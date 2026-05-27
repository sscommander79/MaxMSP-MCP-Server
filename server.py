# server.py
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
import asyncio
import socketio

from typing import Callable, Any
import logging
import uuid
import os
import json

SOCKETIO_SERVER_URL = os.environ.get("SOCKETIO_SERVER_URL", "http://127.0.0.1")
SOCKETIO_SERVER_PORT = os.environ.get("SOCKETIO_SERVER_PORT", "5002")
NAMESPACE = os.environ.get("NAMESPACE", "/mcp")

current_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(current_dir, "docs.json")
with open(docs_path, "r") as f:
    docs = json.load(f)
flattened_docs = {}
for obj_list in docs.values():
    for obj in obj_list:
        flattened_docs[obj["name"]] = obj

io_server_started = False


class MaxMSPConnection:
    def __init__(self, server_url: str, server_port: int, namespace: str = NAMESPACE):

        self.server_url = server_url
        self.server_port = server_port
        self.namespace = namespace

        self.sio = socketio.AsyncClient()
        self._pending = {}  # fetch requests that are not yet completed

        @self.sio.on("response", namespace=self.namespace)
        async def _on_response(data):
            req_id = data.get("request_id")
            fut = self._pending.get(req_id)
            if fut and not fut.done():
                fut.set_result(data.get("results"))

    async def send_command(self, cmd: dict):
        """Send a command to MaxMSP."""
        await self.sio.emit("command", cmd, namespace=self.namespace)
        logging.info(f"Sent to MaxMSP: {cmd}")

    async def send_request(self, payload: dict, timeout=2.0):
        """Send a fetch request to MaxMSP."""
        request_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        payload.update({"request_id": request_id})
        await self.sio.emit("request", payload, namespace=self.namespace)
        logging.info(f"Request to MaxMSP: {payload}")

        try:
            response = await asyncio.wait_for(future, timeout)
            return response
        except asyncio.TimeoutError:
            raise TimeoutError(f"No response received in {timeout} seconds.")
        finally:
            self._pending.pop(request_id, None)

    async def start_server(self) -> None:
        """IMPORTANT: This method should only be called ONCE per application instance.
        Multiple calls can lead to binding multiple ports unnecessarily.
        """
        try:
            # Connect to the server
            full_url = f"{self.server_url}:{self.server_port}"
            await self.sio.connect(full_url, namespaces=self.namespace)
            logging.info(f"Connected to Socket.IO server at {full_url}")
            return

        except OSError as e:
            logging.error(f"Error starting Socket.IO server: {e}")


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Manage server lifespan"""
    global io_server_started
    if not io_server_started:
        try:
            maxmsp = MaxMSPConnection(
                SOCKETIO_SERVER_URL, SOCKETIO_SERVER_PORT, NAMESPACE
            )
            try:
                # Start the Socket.IO server
                await maxmsp.start_server()
                io_server_started = True
                logging.info(f"Listening on {maxmsp.server_url}:{maxmsp.server_port}")

                # Yield the Socket.IO connection to make it available in the lifespan context
                yield {"maxmsp": maxmsp}
            except Exception as e:
                logging.error(f"lifespan error starting server: {e}")
                await maxmsp.sio.disconnect()
                raise

        finally:
            logging.info("Shutting down connection")
            await maxmsp.sio.disconnect()
    else:
        logging.info(
            f"IO server already running on {maxmsp.server_url}:{maxmsp.server_port}"
        )


# Create the MCP server with lifespan support
mcp = FastMCP(
    "MaxMSPMCP",
    description="MaxMSP integration through the Model Context Protocol",
    lifespan=server_lifespan,
)


@mcp.tool()
async def add_max_object(
    ctx: Context,
    position: list,
    obj_type: str,
    varname: str,
    args: list,
):
    """Add a new Max object.

    The position is is a list of two integers representing the x and y coordinates,
    which should be outside the rectangular area returned by get_avoid_rect_position() function.

    Args:
        position (list): Position in the Max patch as [x, y].
        obj_type (str): Type of the Max object (e.g., "cycle~", "dac~").
        varname (str): Variable name for the object.
        args (list): Arguments for the object.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    assert len(position) == 2, "Position must be a list of two integers."
    cmd = {"action": "add_object"}
    kwargs = {
        "position": position,
        "obj_type": obj_type,
        "args": args,
        "varname": varname,
    }
    cmd.update(kwargs)
    await maxmsp.send_command(cmd)


@mcp.tool()
async def remove_max_object(
    ctx: Context,
    varname: str,
):
    """Delete a Max object.

    Args:
        varname (str): Variable name for the object.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    cmd = {"action": "remove_object"}
    kwargs = {"varname": varname}
    cmd.update(kwargs)
    await maxmsp.send_command(cmd)


@mcp.tool()
async def connect_max_objects(
    ctx: Context,
    src_varname: str,
    outlet_idx: int,
    dst_varname: str,
    inlet_idx: int,
):
    """Connect two Max objects.

    Args:
        src_varname (str): Variable name of the source object.
        outlet_idx (int): Outlet index on the source object.
        dst_varname (str): Variable name of the destination object.
        inlet_idx (int): Inlet index on the destination object.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    cmd = {"action": "connect_objects"}
    kwargs = {
        "src_varname": src_varname,
        "outlet_idx": outlet_idx,
        "dst_varname": dst_varname,
        "inlet_idx": inlet_idx,
    }
    cmd.update(kwargs)
    await maxmsp.send_command(cmd)


@mcp.tool()
async def disconnect_max_objects(
    ctx: Context,
    src_varname: str,
    outlet_idx: int,
    dst_varname: str,
    inlet_idx: int,
):
    """Disconnect two Max objects.

    Args:
        src_varname (str): Variable name of the source object.
        outlet_idx (int): Outlet index on the source object.
        dst_varname (str): Variable name of the destination object.
        inlet_idx (int): Inlet index on the destination object.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    cmd = {"action": "disconnect_objects"}
    kwargs = {
        "src_varname": src_varname,
        "outlet_idx": outlet_idx,
        "dst_varname": dst_varname,
        "inlet_idx": inlet_idx,
    }
    cmd.update(kwargs)
    await maxmsp.send_command(cmd)


@mcp.tool()
async def set_object_attribute(
    ctx: Context,
    varname: str,
    attr_name: str,
    attr_value: list,
):
    """Set an attribute of a Max object.

    Args:
        varname (str): Variable name of the object.
        attr_name (str): Name of the attribute to be set.
        attr_value (list): Values of the attribute to be set.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    cmd = {"action": "set_object_attribute"}
    kwargs = {"varname": varname, "attr_name": attr_name, "attr_value": attr_value}
    cmd.update(kwargs)
    await maxmsp.send_command(cmd)


@mcp.tool()
async def set_message_text(
    ctx: Context,
    varname: str,
    text_list: list,
):
    """Set the text of a message object in MaxMSP.

    Args:
        varname (str): Variable name of the message object.
        text_list (list): A list of arguments to be set to the message object.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    cmd = {"action": "set_message_text"}
    kwargs = {"varname": varname, "new_text": text_list}
    cmd.update(kwargs)
    await maxmsp.send_command(cmd)


@mcp.tool()
async def send_bang_to_object(ctx: Context, varname: str):
    """Send a bang to an object in MaxMSP.

    Args:
        varname (str): Variable name of the object to be banged.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    cmd = {"action": "send_bang_to_object"}
    kwargs = {"varname": varname}
    cmd.update(kwargs)
    await maxmsp.send_command(cmd)


@mcp.tool()
async def send_messages_to_object(
    ctx: Context,
    varname: str,
    message: list,
):
    """Send a message to an object in MaxMSP. The message is made of a list of arguments.

    When using message to set attributes, one attribute can only be set by one message.
    For example, to set the "size" attribute of a "button" object, use:
    send_messages_to_object("button1", ["size", 100, 100])
    To set the "size" and "color" attributes of a "button" object, use the tool for two times:
    send_messages_to_object("button1", ["size", 100, 100])
    send_messages_to_object("button1", ["color", 0, 0, 0])

    Args:
        varname (str): Variable name of the object to be messaged.
        message (list): A list of messages to be sent to the object.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    cmd = {"action": "send_message_to_object"}
    kwargs = {"varname": varname, "message": message}
    cmd.update(kwargs)
    await maxmsp.send_command(cmd)


@mcp.tool()
async def set_number(
    ctx: Context,
    varname: str,
    num: float,
):
    """Set the value of a object in MaxMSP.
    The object can be a number box, a slider, a dial, a gain.

    Args:
        varname (str): Variable name of the comment object.
        num (float): Value to be set for the object.
    """

    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    cmd = {"action": "set_number"}
    kwargs = {"varname": varname, "num": num}
    cmd.update(kwargs)
    await maxmsp.send_command(cmd)


@mcp.tool()
def list_all_objects(ctx: Context) -> list:
    """Returns a name list of all objects that can be added in Max.
    To understand a specific object in the list, use the `get_object_doc` tool."""
    return list(flattened_docs.keys())


@mcp.tool()
def get_object_doc(ctx: Context, object_name: str) -> dict:
    """Retrieve the official documentation for a given object.
    Use this resource to understand how a specific object works, including its
    description, inlets, outlets, arguments, methods(messages), and attributes.

    Args:
        object_name (str): Name of the object to look up.

    Returns:
        dict: Official documentations for the specified object.
    """
    try:
        return flattened_docs[object_name]
    except KeyError:
        return {
            "success": False,
            "error": "Invalid object name",
            "suggestion": "Make sure the object name is a valid Max object name.",
        }


@mcp.tool()
async def get_objects_in_patch(
    ctx: Context,
):
    """Retrieve the list of existing objects in the current Max patch.

    Use this to understand the current state of the patch, including the
    objects(boxes) and patch cords(lines). The retrieved list contains a
    list of objects including their maxclass, varname for scripting,
    position(patching_rect), and the boxtext when available, as well as a
    list of patch cords with their source and destination information.

    Returns:
        list: A list of objects and patch cords.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    payload = {"action": "get_objects_in_patch"}
    response = await maxmsp.send_request(payload)

    return [response]


@mcp.tool()
async def get_objects_in_selected(
    ctx: Context,
):
    """Retrieve the list of objects that is selected in a (unlocked) patcher window.

    Use this when the user wanted to reference to the selected objects.

    Returns:
        list: A list of objects and patch cords.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    payload = {"action": "get_objects_in_selected"}
    response = await maxmsp.send_request(payload)

    return [response]


@mcp.tool()
async def get_object_attributes(ctx: Context, varname: str):
    """Retrieve an objects' attributes and values of the attributes.

    Use this to understand the state of an object.

    Returns:
        list: A list of attributes name and attributes values.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    payload = {"action": "get_object_attributes"}
    kwargs = {"varname": varname}
    payload.update(kwargs)
    response = await maxmsp.send_request(payload)

    return [response]


@mcp.tool()
async def get_avoid_rect_position(ctx: Context):
    """When deciding the position to add a new object to the path, this rectangular area
    should be avoid. This is useful when you want to add an object to the patch without
    overlapping with existing objects.

    Returns:
        list: A list of four numbers representing the left, top, right, bottom of the rectangular area.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    payload = {"action": "get_avoid_rect_position"}
    response = await maxmsp.send_request(payload)

    return response


@mcp.tool()
def query_maxmsp_docs(ctx: Context, question: str) -> str:
    """Search the Max/MSP reference library and get an expert answer.
    
    Use this to look up how to build anything in Max/MSP, understand objects,
    find best practices, or get sound design advice. The library contains:
    - Complete Cycling '74 object reference (1,174 objects with inlets/outlets/attributes)
    - All Max/MSP tutorials and user guide
    - Sound design theory (analog synthesis, classic synths, FM, granular, etc.)
    - Real patch examples from Fors, K Devices, Envelop, user patches
    - gen~ DSP documentation
    - Max for Live / Live Object Model documentation
    
    Args:
        question: Any question about Max/MSP patching, objects, sound design, or best practices
    
    Returns:
        Detailed expert answer with patch diagrams and step-by-step instructions
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Library/Application Support/Genspark Claw/users/abdc6d4c-bc92-4faf-b07d-db6fe61304ea/workspace/max-rag'))
    
    # Use the RAG query function directly
    WORKSPACE = os.path.expanduser("~/Library/Application Support/Genspark Claw/users/abdc6d4c-bc92-4faf-b07d-db6fe61304ea/workspace")
    CHROMA_PATH = os.path.join(WORKSPACE, "max-rag/chroma_db")
    
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from openai import OpenAI
        import re
        
        db_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = db_client.get_collection("maxmsp")
        
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        q_embedding = embed_model.encode(question).tolist()
        
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=8,
            include=["documents", "metadatas"]
        )
        
        chunks = results["documents"][0]
        metas = results["metadatas"][0]
        context = "\n\n---\n\n".join([
            f"[{m.get('topic','?')}]\n{chunk}"
            for chunk, m in zip(chunks, metas)
        ])
        
        # Get API key from OpenClaw config
        config_path = os.path.expanduser(
            "~/Library/Application Support/Genspark Claw/users/abdc6d4c-bc92-4faf-b07d-db6fe61304ea/openclaw.json"
        )
        api_key = None
        try:
            with open(config_path) as f:
                content = f.read()
            match = re.search(r'"apiKey":\s*"([^"]+)"', content)
            if match:
                api_key = match.group(1)
        except Exception:
            pass
        
        if not api_key:
            return "Error: Could not read API key from OpenClaw config"
        
        client_ai = OpenAI(api_key=api_key, base_url="https://www.genspark.ai/api/llm_proxy/v1")
        
        response = client_ai.chat.completions.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[
                {"role": "system", "content": """You are an expert Max/MSP developer. Give clear, specific answers with exact object names, inlet/outlet numbers, and working patch examples. Plain English first, technical terms second. Always include a patch diagram in a code block."""},
                {"role": "user", "content": f"Reference material:\n\n{context}\n\n---\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"RAG query error: {e}"


@mcp.tool()
def lookup_max_object_reference(ctx: Context, object_name: str) -> str:
    """Look up the complete reference for any Max/MSP object.
    
    Returns full inlets, outlets, arguments, attributes, and usage examples
    for any Max, MSP, Jitter, or M4L object.
    
    Args:
        object_name: The object name (e.g. 'cycle~', 'groove~', 'jit.matrix', 'live.object')
    
    Returns:
        Complete object reference including all inlets, outlets, arguments, attributes
    """
    import sys, os, re
    WORKSPACE = os.path.expanduser("~/Library/Application Support/Genspark Claw/users/abdc6d4c-bc92-4faf-b07d-db6fe61304ea/workspace")
    CHROMA_PATH = os.path.join(WORKSPACE, "max-rag/chroma_db")
    
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        db_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = db_client.get_collection("maxmsp")
        
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        q = f"OBJECT REFERENCE: {object_name} inlets outlets arguments attributes"
        q_embedding = embed_model.encode(q).tolist()
        
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=5,
            include=["documents", "metadatas"],
            where={"topic": {"$in": ["Max Objects", "MSP Audio Objects", "Jitter Objects", "M4L Objects", "UI Objects", "MSP Synthesis Objects"]}}
        )
        
        if not results["documents"][0]:
            # Fallback without filter
            results = collection.query(
                query_embeddings=[q_embedding],
                n_results=5,
                include=["documents", "metadatas"]
            )
        
        docs = results["documents"][0]
        # Find the most relevant chunk (one that contains the object name)
        best = "\n\n".join([d for d in docs if object_name in d] or docs[:3])
        return best if best else f"No reference found for '{object_name}'"
        
    except Exception as e:
        return f"Lookup error: {e}"


if __name__ == "__main__":
    mcp.run()
