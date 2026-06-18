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
    """Manage server lifespan — Max connection is optional.
    RAG tools work without Max open; patching tools need Max running."""
    global io_server_started
    maxmsp = MaxMSPConnection(SOCKETIO_SERVER_URL, SOCKETIO_SERVER_PORT, NAMESPACE)

    if not io_server_started:
        try:
            await maxmsp.start_server()
            io_server_started = True
            logging.info(f"✅ Max connected on {maxmsp.server_url}:{maxmsp.server_port}")
        except Exception as e:
            # Max not open — RAG tools still work, patching tools won't
            logging.warning(
                f"Max not connected (port {SOCKETIO_SERVER_PORT} unavailable) — "
                f"RAG query tools active, patching tools inactive. Open Max to enable patching."
            )
            # Don't raise — yield anyway so RAG tools are available
    else:
        logging.info(f"IO server already running on {maxmsp.server_url}:{maxmsp.server_port}")

    try:
        yield {"maxmsp": maxmsp}
    finally:
        logging.info("Shutting down connection")
        try:
            await maxmsp.sio.disconnect()
        except Exception:
            pass


# Create the MCP server with lifespan support
mcp = FastMCP(
    "MaxMSPMCP",
    instructions="MaxMSP integration through the Model Context Protocol",
    lifespan=server_lifespan,
)

# ── RAG: module-level singletons (loaded once, reused on every query) ──────────
import os as _os, re as _re
_WORKSPACE   = _os.path.expanduser("~/Desktop/AI/Max MSP Training Tool/MaxMSP-Corpus")
_CHROMA_PATH = _os.path.expanduser("~/Desktop/AI/Max MSP Training Tool/MaxMSP-RAG/chroma_db")
_rag_collection  = None   # lazy-loaded on first query
_rag_embed_model = None   # lazy-loaded on first query

def _get_rag():
    """Return (collection, embed_model) — initialised once, cached forever."""
    global _rag_collection, _rag_embed_model
    if _rag_collection is None:
        import chromadb
        from sentence_transformers import SentenceTransformer
        _db = chromadb.PersistentClient(path=_CHROMA_PATH)
        _rag_collection  = _db.get_collection("maxmsp")
        _rag_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _rag_collection, _rag_embed_model

# Actual topic labels used in the DB (from audit 2026-05-30)
_OBJECT_REF_TOPICS = [
    "Object Reference", "MSP Audio", "MSP (Audio)", "MSP Synthesis Objects",
    "Jitter Video", "Jitter (Video)", "JavaScript", "UI Objects",
    "Max Basics", "MAX (Core)", "Max for Live", "M4L (Live)", "RNBO",
]


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
    try:
        from openai import OpenAI
        import re

        collection, embed_model = _get_rag()
        q_embedding = embed_model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=12,
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
    try:
        collection, embed_model = _get_rag()
        q = f"OBJECT REFERENCE: {object_name} inlets outlets arguments attributes"
        q_embedding = embed_model.encode(q).tolist()

        # First pass: filter to known reference topics (actual DB labels)
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=6,
            include=["documents", "metadatas"],
            where={"topic": {"$in": _OBJECT_REF_TOPICS}}
        )

        docs = results["documents"][0]

        # If nothing matched the filter, fall back to unfiltered
        if not docs:
            results = collection.query(
                query_embeddings=[q_embedding],
                n_results=6,
                include=["documents", "metadatas"]
            )
            docs = results["documents"][0]

        # Prefer chunks that literally contain the object name
        matched = [d for d in docs if object_name in d]
        best = "\n\n---\n\n".join(matched if matched else docs[:3])
        return best if best else f"No reference found for '{object_name}'"

    except Exception as e:
        return f"Lookup error: {e}"


# ── Higher-level build + verify (graph-in, auto-layout, read-back self-heal) ────

def _norm_dump(response):
    """Normalize a get_objects_in_patch response into (varname->maxclass, cordset).

    cordset is a set of (src_varname, src_outlet, dst_varname, dst_inlet).
    Tolerant of the response arriving as a dict or a JSON string.
    """
    import json as _json
    data = response
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, str):
        try:
            data = _json.loads(data)
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    boxes = {}
    for b in data.get("boxes", []) or []:
        box = b.get("box", b) if isinstance(b, dict) else {}
        vn = box.get("varname")
        if vn:
            boxes[vn] = box.get("maxclass")
    cords = set()
    for l in data.get("lines", []) or []:
        pl = l.get("patchline", l) if isinstance(l, dict) else {}
        src = pl.get("source") or []
        dst = pl.get("destination") or []
        if len(src) == 2 and len(dst) == 2:
            cords.add((src[0], int(src[1]), dst[0], int(dst[1])))
    return boxes, cords


def _auto_layout(objects, connections, origin, col_w, row_h):
    """Assign [x,y] per object id by longest-path layering (top-to-bottom flow).
    Comments are pushed to a right-side lane so cords never cross them.
    Returns {id: [x, y]}.
    """
    ids = [o["id"] for o in objects]
    is_comment = {o["id"]: (o.get("type") == "comment") for o in objects}
    succ = {i: [] for i in ids}
    for c in connections:
        f, t = c.get("from"), c.get("to")
        if f in succ and t in succ:
            succ[f].append(t)
    layer = {i: 0 for i in ids}
    # relax longest-path; cap iterations to survive accidental cycles
    for _ in range(len(ids) + 1):
        changed = False
        for f in ids:
            for t in succ[f]:
                if layer[t] < layer[f] + 1:
                    layer[t] = layer[f] + 1
                    changed = True
        if not changed:
            break
    ox, oy = origin
    by_layer = {}
    flow_ids = [i for i in ids if not is_comment[i]]
    for i in flow_ids:
        by_layer.setdefault(layer[i], []).append(i)
    pos = {}
    for lyr, members in by_layer.items():
        for col, i in enumerate(members):
            pos[i] = [ox + col * col_w, oy + lyr * row_h]
    max_cols = max((len(m) for m in by_layer.values()), default=1)
    lane_x = ox + max_cols * col_w + 80
    crow = 0
    for i in ids:
        if is_comment[i]:
            pos[i] = [lane_x, oy + crow * 40]
            crow += 1
    return pos


@mcp.tool()
async def build_patch(
    ctx: Context,
    objects: list,
    connections: list,
    origin: list = [40, 40],
):
    """Build a whole Max patch from a declarative graph in one call, then read it
    back and report anything that did not take. Preferred for building from scratch.

    The model describes the graph; the server lays it out (top-to-bottom flow,
    parallel nodes side by side, comments in a right-side lane), wires everything by
    id, then dumps the live patch and reports any missing object or connection —
    nothing fails silently. It also self-heals: missing connections are retried once.

    Args:
        objects (list): each {"id": str (unique, used as the scripting varname),
            "type": str (maxclass, e.g. "cycle~", "*~", "dac~", "message", "comment",
            "toggle", "number", "slider", "button"),
            "args": list (optional; object arguments. For "message"/"comment"/"flonum"
            this is the text/content)}.
        connections (list): each {"from": id, "to": id, "outlet": int=0, "inlet": int=0}.
        origin (list): [x, y] top-left anchor for layout. Default [40, 40].

    Returns:
        dict: report with created count, objects_missing, connections_total,
        connections_missing (after one self-heal pass), and healed.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    if maxmsp is None or not getattr(maxmsp.sio, "connected", False):
        return {"success": False, "error": "Max is not connected. Open the host patch "
                "(MaxMSP_Agent) so the bridge is live, then retry."}

    # validate ids
    ids = [o.get("id") for o in objects]
    if len(set(ids)) != len(ids) or None in ids:
        return {"success": False, "error": "Each object needs a unique non-null 'id'."}

    pos = _auto_layout(objects, connections, origin, col_w=150, row_h=70)

    # 1) create objects
    for o in objects:
        oid, otype = o["id"], o.get("type", "")
        args = o.get("args", []) or []
        await maxmsp.send_command({
            "action": "add_object", "position": pos[oid],
            "obj_type": otype, "args": args, "varname": oid,
        })
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.4)

    # 2) wire connections
    intended = []
    for c in connections:
        f, t = c.get("from"), c.get("to")
        ci, co = int(c.get("outlet", 0)), int(c.get("inlet", 0))
        intended.append((f, ci, t, co))
        await maxmsp.send_command({
            "action": "connect_objects", "src_varname": f, "outlet_idx": ci,
            "dst_varname": t, "inlet_idx": co,
        })
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.4)

    # 3) read back + diff
    try:
        dump = await maxmsp.send_request({"action": "get_objects_in_patch"}, timeout=4.0)
    except Exception as e:
        return {"success": False, "error": f"Built, but read-back failed: {e}",
                "note": "Objects/cords were sent; could not verify."}
    present, cords = _norm_dump(dump)
    missing_objs = [oid for oid in ids if oid not in present]
    missing_cords = [c for c in intended if c not in cords]

    # 4) self-heal missing connections once
    healed = []
    if missing_cords:
        for (f, ci, t, co) in missing_cords:
            await maxmsp.send_command({
                "action": "connect_objects", "src_varname": f, "outlet_idx": ci,
                "dst_varname": t, "inlet_idx": co,
            })
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.4)
        try:
            dump = await maxmsp.send_request({"action": "get_objects_in_patch"}, timeout=4.0)
            present, cords = _norm_dump(dump)
            still = [c for c in intended if c not in cords]
            healed = [c for c in missing_cords if c not in still]
            missing_cords = still
            missing_objs = [oid for oid in ids if oid not in present]
        except Exception:
            pass

    return {
        "success": len(missing_objs) == 0 and len(missing_cords) == 0,
        "objects_created": len(ids) - len(missing_objs),
        "objects_total": len(ids),
        "objects_missing": missing_objs,
        "connections_total": len(intended),
        "connections_missing": [
            {"from": f, "outlet": ci, "to": t, "inlet": co}
            for (f, ci, t, co) in missing_cords
        ],
        "connections_healed": len(healed),
        "hint": ("All objects and connections verified present."
                 if not missing_objs and not missing_cords else
                 "Some items missing after self-heal — check object types/ids and inlet/outlet indices."),
    }


@mcp.tool()
async def verify_patch(
    ctx: Context,
    objects: list = [],
    connections: list = [],
):
    """Read the live patch back and diff it against an intended graph. Read-only.

    Use after building (by any means — build_patch, the unitary tools, or a patch
    emitted by your own builder like MaxPyLang and opened in Max) to confirm the live
    patch matches intent. Reports what is missing and what is unexpected (e.g. objects
    the user added by hand). This is the round-trip correctness check.

    Args:
        objects (list): expected objects, each {"id": varname, "type": maxclass (optional)}.
        connections (list): expected {"from": id, "to": id, "outlet": int=0, "inlet": int=0}.

    Returns:
        dict: objects_missing, objects_unexpected, connections_missing,
        connections_unexpected, plus counts. Plumbing (maxmcp*) is ignored.
    """
    maxmsp = ctx.request_context.lifespan_context.get("maxmsp")
    if maxmsp is None or not getattr(maxmsp.sio, "connected", False):
        return {"success": False, "error": "Max is not connected. Open the host patch first."}
    try:
        dump = await maxmsp.send_request({"action": "get_objects_in_patch"}, timeout=4.0)
    except Exception as e:
        return {"success": False, "error": f"Read-back failed: {e}"}
    present, cords = _norm_dump(dump)
    present_live = {v: m for v, m in present.items() if not str(v).startswith("maxmcp")}

    want_ids = [o.get("id") for o in objects]
    intended_cords = set()
    for c in connections:
        intended_cords.add((c.get("from"), int(c.get("outlet", 0)),
                            c.get("to"), int(c.get("inlet", 0))))

    objects_missing = [i for i in want_ids if i not in present_live]
    objects_unexpected = [v for v in present_live if v not in want_ids] if want_ids else []
    conns_missing = [
        {"from": f, "outlet": ci, "to": t, "inlet": co}
        for (f, ci, t, co) in intended_cords if (f, ci, t, co) not in cords
    ]
    conns_unexpected = (
        [{"from": f, "outlet": ci, "to": t, "inlet": co}
         for (f, ci, t, co) in cords if (f, ci, t, co) not in intended_cords]
        if intended_cords else []
    )
    return {
        "success": not objects_missing and not conns_missing,
        "objects_in_patch": len(present_live),
        "objects_missing": objects_missing,
        "objects_unexpected": objects_unexpected,
        "connections_in_patch": len(cords),
        "connections_missing": conns_missing,
        "connections_unexpected": conns_unexpected,
    }


if __name__ == "__main__":
    mcp.run()
