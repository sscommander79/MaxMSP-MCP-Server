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
_HERE        = _os.path.dirname(_os.path.abspath(__file__))
_WORKSPACE   = _os.environ.get("MAXMSP_CORPUS_DIR",  _os.path.expanduser("~/Desktop/AI/maxmsp-toolkit/maxmsp-corpus/licensed/MaxMSP-Corpus"))
_CHROMA_PATH = _os.environ.get("MAXMSP_CHROMA_PATH", _os.path.normpath(_os.path.join(_HERE, "..", "maxmsp-reference-library", "chroma_db")))

# ── Cross-repo import: retrieval.py lives in the sibling reference-library ──
import sys as _sys
_REF_LIB_PATH = _os.environ.get(
    "MAXMSP_REF_LIB_PATH",
    _os.path.normpath(_os.path.join(_os.path.dirname(__file__), "..", "maxmsp-reference-library"))
)
if _REF_LIB_PATH not in _sys.path:
    _sys.path.insert(0, _REF_LIB_PATH)
from retrieval import retrieve as _retrieve, build_bm25_index as _build_bm25_index
from query import _get_reranker, _classify_tier
from objectdb import check_object_names_in_query
from validator import validate_answer_objects

_rag_collection  = None   # lazy-loaded on first query
_rag_embed_model = None   # lazy-loaded on first query
_rag_bm25        = None   # lazy — BM25Okapi over full corpus; built once
_rag_corpus_ids  = None   # lazy — parallel ID list for BM25 index

def _get_rag():
    """Return (collection, embed_model, bm25, corpus_ids) — initialised once, cached forever."""
    global _rag_collection, _rag_embed_model, _rag_bm25, _rag_corpus_ids
    if _rag_collection is None:
        import chromadb
        from sentence_transformers import SentenceTransformer
        _db = chromadb.PersistentClient(path=_CHROMA_PATH)
        _rag_collection  = _db.get_collection("maxmsp")
        _rag_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _rag_bm25, _rag_corpus_ids = _build_bm25_index(_rag_collection)
        # BM25 index is built at startup — restart after ingest.
    return _rag_collection, _rag_embed_model, _rag_bm25, _rag_corpus_ids

# ── LLM provider config (ADR-0003): env-var-only, BYOK-ready ──
_LLM_BASE_URL   = _os.environ.get("LLM_BASE_URL", "https://www.genspark.ai/api/llm_proxy/v1")
_LLM_MODEL      = _os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
_LLM_FAST_MODEL = _os.environ.get("LLM_FAST_MODEL", "")   # empty = use _LLM_MODEL for all tiers
# Retrieval-confidence gate (cosine distance; lower = closer). Env-tunable.
# Calibration (all-MiniLM-L6-v2): covered Max questions ~0.33-0.55, off-topic ~0.70.
# The grounded system prompt is the PRIMARY guard (catches fabricated objects whose
# text isn't in context); this gate is a coarse backstop tuned to avoid falsely
# refusing valid-but-thin Max questions.
_RAG_WEAK_DIST    = float(_os.environ.get("RAG_WEAK_DIST", "0.85"))    # hard-refuse above this
_RAG_CAUTION_DIST = float(_os.environ.get("RAG_CAUTION_DIST", "0.6"))   # low-confidence note above this


def _get_llm_key():
    """Resolve the generation API key from env vars only (ADR-0003).
    Returns '' if none.
    NOTE: GUI-launched apps don't source ~/.zshenv, so set the key in the MCP
    connector env (claude_desktop_config.json / ~/.codex/config.toml)."""
    for var in ("GENSPARK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"):
        v = _os.environ.get(var)
        if v:
            return v
    print("No API key found; set GENSPARK_API_KEY for generation. Retrieval works without a key.", file=_sys.stderr)
    return ""

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

        collection, embed_model, bm25, corpus_ids = _get_rag()

        # ── Gate A: fabricated-object pre-check (ANS-04) ─────────────────────────
        unknown_objs = check_object_names_in_query(question)
        if unknown_objs:
            return (
                f"The object(s) {unknown_objs} do not appear in the Max/MSP object database. "
                "This may be a fabricated or misspelled name. Check the object name and try again, "
                "or use lookup_max_object_reference to confirm the correct name."
            )

        result = _retrieve(
            question,
            collection,
            embed_model,
            bm25,
            corpus_ids,
            n_results=12,
            weak_dist=_RAG_WEAK_DIST,
            caution_dist=_RAG_CAUTION_DIST,
            reranker=_get_reranker(),
        )
        if result["refused"]:
            return result["refusal_msg"]
        chunks = result["chunks"]
        metas = result["metas"]
        low_conf = result["low_conf"]
        context = result["context_str"]
        used_sources = sorted({m.get("source", "?") for m in metas})

        # Retrieval boost: if the question names known Max objects, prepend their
        # authoritative docs.json reference so the answer uses exact I/O specs rather
        # than only fuzzy chunks. Bridges the object database into the Q&A path.
        for obj in _objects_in_question(question):
            context = ("[AUTHORITATIVE OBJECT REFERENCE]\n" + _format_object_doc(obj)
                       + "\n\n---\n\n" + context)
            used_sources.append(obj["name"] + " (object reference)")
        used_sources = sorted(set(used_sources))

        api_key = _get_llm_key()
        if not api_key:
            return ("Error: no LLM API key found. Set GENSPARK_API_KEY in the MCP "
                    "connector env (claude_desktop_config.json / ~/.codex/config.toml), "
                    "or sign in to Genspark. (Retrieval works without a key; only the "
                    "generated answer needs one.)")

        client_ai = OpenAI(api_key=api_key, base_url=_LLM_BASE_URL, timeout=120.0)

        system_prompt = (
            "You are an expert Max/MSP teacher. Answer ONLY from the reference material "
            "provided in the user message. Hard rules:\n"
            "1. Ground every factual claim in the provided material. Do NOT use outside "
            "knowledge for object names, inlets, outlets, arguments, or attributes.\n"
            "2. If the material does not contain the answer, say so plainly (\"the "
            "reference library doesn't cover this clearly\") — do NOT invent object "
            "names, inlet/outlet numbers, or arguments to fill the gap.\n"
            "3. If unsure of an exact object spec, tell the user to confirm with the "
            "lookup_max_object_reference tool instead of guessing.\n"
            "4. Teach plainly: plain English first, technical term second; include a "
            "patch diagram in a code block when you have enough grounded detail.\n"
            "5. End with a 'Sources:' line listing the source files you actually used."
        )
        if low_conf:
            system_prompt += ("\nNOTE: retrieval confidence is LOW here — be especially "
                              "cautious and explicit about what the material does not cover.")

        user_message = (f"Reference material:\n\n{context}\n\n---\n\nQuestion: {question}\n\n"
                        f"(Source files available to cite: {', '.join(used_sources)})")

        # ── SPD-05: model tiering ─────────────────────────────────────────────────
        tier = _classify_tier(question)
        model_to_use = (_LLM_FAST_MODEL or _LLM_MODEL) if tier == "fast" else _LLM_MODEL

        response = client_ai.chat.completions.create(
            model=model_to_use,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        answer = response.choices[0].message.content

        # ── Gate B: post-generation validator (ANS-04) ────────────────────────────
        _validation = validate_answer_objects(answer)
        if not _validation["ok"]:
            answer += _validation["warning"]

        # Safety net: always surface the grounding sources even if the model omits them.
        if "Sources:" not in answer:
            answer += "\n\nSources: " + ", ".join(used_sources)
        return answer

    except Exception as e:
        return f"RAG query error: {e}"


def _resolve_object(name: str):
    """Resolve an object name to its docs.json entry, tolerant of case and the
    trailing '~'. Exact match wins; then case-insensitive; then add/strip '~'."""
    if not name:
        return None
    if name in flattened_docs:
        return flattened_docs[name]
    low = name.lower()
    for k, v in flattened_docs.items():
        if k.lower() == low:
            return v
    if (name + "~") in flattened_docs:
        return flattened_docs[name + "~"]
    if name.endswith("~") and name[:-1] in flattened_docs:
        return flattened_docs[name[:-1]]
    return None


def _format_object_doc(obj: dict) -> str:
    """Render a docs.json object entry as a readable reference block."""
    out = [f"OBJECT REFERENCE: {obj.get('name', '?')}"]
    if obj.get("digest"):
        out.append(f"Digest: {obj['digest']}")
    if obj.get("description"):
        out.append(f"\nDESCRIPTION:\n{obj['description']}")

    def _section(title, items, fmt):
        if not items:
            return
        out.append(f"\n{title}:")
        for it in items:
            out.append(fmt(it))

    _section("INLETS", obj.get("inletlist"),
             lambda i: f"- Inlet {i.get('id', '?')} ({i.get('type', '')}): {i.get('digest', '')}".rstrip())
    _section("OUTLETS", obj.get("outletlist"),
             lambda o: f"- Outlet {o.get('id', '?')} ({o.get('type', '')}): {o.get('digest', '')}".rstrip())
    _section("ARGUMENTS", obj.get("arguments"),
             lambda a: f"- {a.get('name', '?')} ({a.get('type', '')}, "
                       f"{'required' if str(a.get('optional')) == '0' else 'optional'}): {a.get('digest', '')}".rstrip())
    _section("MESSAGES", obj.get("methods"),
             lambda m: f"- {m.get('name', '?')}: {m.get('digest', '')}".rstrip())
    _section("ATTRIBUTES", obj.get("attributes"),
             lambda a: f"- {a.get('name', '?')} ({a.get('type', '')}): {a.get('digest', '')}".rstrip())
    return "\n".join(out)


def _objects_in_question(question, limit=2):
    """Known Max object names explicitly mentioned in the question. Object-like
    tokens only (end with '~', contain '.', or len>=5) so we don't match common
    English words that happen to be object names."""
    found, seen = [], set()
    for t in sorted(set(_re.findall(r"[A-Za-z][A-Za-z0-9_.]*~?", question or "")),
                    key=len, reverse=True):
        if not (t.endswith("~") or "." in t or len(t) >= 5):
            continue
        obj = _resolve_object(t)
        if obj and obj["name"] not in seen:
            seen.add(obj["name"])
            found.append(obj)
        if len(found) >= limit:
            break
    return found


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
    # 1) Authoritative exact lookup from docs.json (official object database).
    #    Deterministic — no semantic guessing. Tolerant of case and '~' variants.
    obj = _resolve_object(object_name)
    if obj is not None:
        return _format_object_doc(obj)

    # 2) Fallback for objects NOT in the database (e.g. third-party externals):
    #    semantic search, but ONLY return chunks that actually mention the object —
    #    never silently return unrelated objects.
    try:
        collection, embed_model, _bm25, _corpus_ids = _get_rag()
        q = f"OBJECT REFERENCE: {object_name} inlets outlets arguments attributes"
        q_embedding = embed_model.encode(q).tolist()

        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=6,
            include=["documents", "metadatas"],
            where={"topic": {"$in": _OBJECT_REF_TOPICS}}
        )
        docs = results["documents"][0]
        if not docs:
            results = collection.query(
                query_embeddings=[q_embedding],
                n_results=6,
                include=["documents", "metadatas"]
            )
            docs = results["documents"][0]

        matched = [d for d in docs if object_name in d]
        if matched:
            return "\n\n---\n\n".join(matched)
        return (f"No exact reference found for '{object_name}'. It is not in the "
                f"built-in Max object database — check the spelling, or call "
                f"list_all_objects to see valid names.")

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


# Maxclasses that are valid but not keyed in the object database (UI / literal boxes).
_KNOWN_UI_TYPES = {
    "message", "comment", "toggle", "number", "flonum", "int", "intbox", "slider",
    "button", "panel", "bpatcher", "subpatcher", "patcher", "dial", "umenu", "preset",
    "live.dial", "live.slider", "live.toggle", "live.button", "live.numbox", "live.text",
    "live.menu", "live.tab", "live.grid", "textedit", "led", "matrixctrl",
}


def _validate_graph(objects, connections):
    """Pre-flight a {objects, connections} graph against the object database. Pure
    (no Max). Catches fabricated object types, duplicate/undefined ids, and
    out-of-range outlet/inlet indices. Errors should block a build; warnings are
    advisory (e.g. arg-determined inlet counts can legitimately exceed the base)."""
    errors, warnings = [], []
    id_type = {}
    all_ids = [o.get("id") for o in objects]
    for dup in sorted({i for i in all_ids if i is not None and all_ids.count(i) > 1}):
        errors.append(f"duplicate object id: '{dup}'")
    for o in objects:
        oid, t = o.get("id"), (o.get("type") or "").strip()
        if not oid:
            errors.append(f"object missing 'id': {o}")
            continue
        id_type[oid] = t
        if not t:
            errors.append(f"object '{oid}' has no type")
        elif t not in _KNOWN_UI_TYPES and _resolve_object(t) is None:
            warnings.append(f"object '{oid}': type '{t}' is not in the object database "
                            "(possible typo, fabricated object, or third-party external)")
    for c in connections:
        f, t = c.get("from"), c.get("to")
        co, ci = int(c.get("outlet", 0)), int(c.get("inlet", 0))
        if f not in id_type:
            errors.append(f"connection from undefined id '{f}'")
            continue
        if t not in id_type:
            errors.append(f"connection to undefined id '{t}'")
            continue
        of, ot = _resolve_object(id_type[f]), _resolve_object(id_type[t])
        if of and of.get("outletlist") and co >= len(of["outletlist"]):
            warnings.append(f"'{f}' ({id_type[f]}) outlet {co} may be out of range "
                            f"(reference shows {len(of['outletlist'])} outlet(s))")
        if ot and ot.get("inletlist") and ci >= len(ot["inletlist"]):
            warnings.append(f"'{t}' ({id_type[t]}) inlet {ci} may be out of range "
                            f"(reference shows {len(ot['inletlist'])}; some objects add inlets via args)")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


@mcp.tool()
def validate_patch_graph(ctx: Context, objects: list, connections: list = []) -> dict:
    """Pre-flight a patch graph against the object database BEFORE building — no Max
    needed. Catches fabricated/misspelled object types, duplicate or undefined ids,
    and out-of-range inlet/outlet indices. Use this to self-check a generated graph,
    fix any errors, then pass the SAME graph to build_patch.

    Args:
        objects (list): each {"id": str, "type": maxclass, "args": list (optional)}.
        connections (list): each {"from": id, "to": id, "outlet": int=0, "inlet": int=0}.

    Returns:
        dict: {"ok": bool, "errors": [...], "warnings": [...]}. Build only when ok.
    """
    return _validate_graph(objects, connections or [])


_AUDIO_OUT_OBJS = {"dac~", "ezdac~", "plugout~", "mc.dac~", "mc.ezdac~", "ezadc~"}
_NO_WIRE_OK = {"comment", "message", "toggle", "button", "number", "flonum", "int",
               "slider", "panel", "comment", "bpatcher", "umenu", "preset"}


def _debug_graph(objects, connections, problem=""):
    """Deterministic, grounded patch diagnosis against the object database. Returns
    a findings list — each {severity, issue, evidence, why, fix}. No Max, no LLM."""
    findings = []

    def add(sev, issue, why, fix, evidence=""):
        findings.append({"severity": sev, "issue": issue, "evidence": evidence,
                         "why": why, "fix": fix})

    val = _validate_graph(objects, connections)
    for e in val["errors"]:
        add("error", e, "the patch graph itself is malformed", "correct the id / type / connection")
    for w in val["warnings"]:
        add("warning", w, "object or inlet/outlet index not confirmed in the object database",
            "verify the object name and the inlet/outlet number")

    id_type = {o.get("id"): (o.get("type") or "").strip() for o in objects}
    out_edges = {o.get("id"): [] for o in objects}
    in_edges = {o.get("id"): [] for o in objects}
    for c in connections:
        if c.get("from") in out_edges:
            out_edges[c["from"]].append(c)
        if c.get("to") in in_edges:
            in_edges[c["to"]].append(c)

    def _xtype(lst, i):
        return (lst[i].get("type", "") if lst and i < len(lst) else "").lower()

    for oid, t in id_type.items():
        obj = _resolve_object(t)
        ins, outs = in_edges.get(oid, []), out_edges.get(oid, [])
        specific = False

        # Audio output with nothing feeding it -> silence.
        if t in _AUDIO_OUT_OBJS and not ins:
            add("error", f"'{oid}' ({t}) is the audio output but nothing is connected to it",
                "with no signal reaching the output you get silence",
                "connect a signal chain (e.g. oscillator -> [*~ <gain>]) into its left inlet")
            specific = True

        # A signal source whose outlet goes nowhere.
        if obj and obj.get("outletlist") and t not in _AUDIO_OUT_OBJS and not outs:
            if any("signal" in (x.get("type", "").lower()) for x in obj["outletlist"]):
                add("warning", f"'{oid}' ({t}) produces a signal but its outlet isn't connected",
                    "its output goes nowhere, so it has no audible effect",
                    "wire its left outlet onward toward the output")
                specific = True

        # Fully isolated object (excluding UI/literal boxes, and anything already
        # flagged more specifically above).
        if not ins and not outs and t not in _NO_WIRE_OK and not specific:
            add("info", f"'{oid}' ({t}) has no connections at all",
                "it is isolated and plays no part in the data/signal flow",
                "wire it in, or remove it if unused")

        # Hot/cold: receives only on cold inlets, never the hot (left) inlet 0.
        if ins and obj and len(obj.get("inletlist", [])) > 1:
            used = {int(c.get("inlet", 0)) for c in ins}
            if 0 not in used:
                add("warning",
                    f"'{oid}' ({t}) only receives on a cold inlet (inlet {min(used)}), never the hot left inlet",
                    "on most objects only the left (hot) inlet triggers output; cold inlets just store "
                    "values — so this object may never fire",
                    "send a bang/value to its left inlet to trigger output (Max evaluates right-to-left, "
                    "so cold inlets are set first)")

    # Signal -> clearly-control inlet mismatch (only when types are unambiguous).
    for c in connections:
        f, t = c.get("from"), c.get("to")
        of, ot = _resolve_object(id_type.get(f, "")), _resolve_object(id_type.get(t, ""))
        if not (of and ot):
            continue
        st_ = _xtype(of.get("outletlist", []), int(c.get("outlet", 0)))
        dt = _xtype(ot.get("inletlist", []), int(c.get("inlet", 0)))
        if "signal" in st_ and dt in ("float", "int", "number"):
            add("warning",
                f"signal outlet of '{f}' ({id_type.get(f)}) feeds a control inlet of '{t}' ({id_type.get(t)})",
                "an audio signal (~) is going into a number/control inlet that doesn't take audio",
                "to read a signal as numbers use [snapshot~] or [number~]; to keep it audio, target a signal inlet",
                evidence=f"{f} outlet {c.get('outlet',0)} -> {t} inlet {c.get('inlet',0)}")

    n_err = sum(1 for x in findings if x["severity"] == "error")
    n_warn = sum(1 for x in findings if x["severity"] == "warning")
    n_info = sum(1 for x in findings if x["severity"] == "info")
    return {
        "problem": problem,
        "summary": (f"{n_err} error(s), {n_warn} warning(s), {n_info} note(s). "
                    + ("No structural problems detected in the graph."
                       if n_err == 0 and n_warn == 0 else
                       "Address errors first, then warnings.")),
        "findings": findings,
    }


@mcp.tool()
def debug_patch(ctx: Context, objects: list, connections: list = [], problem: str = "") -> dict:
    """Diagnose why a Max patch may not be working — grounded in the object database,
    no guessing. Pass the patch as a declarative graph (same shape as build_patch);
    optionally describe the symptom in 'problem'. Deterministic checks: structural
    errors, an audio output with nothing feeding it, signal sources whose outlet is
    unconnected, fully isolated objects, hot/cold inlet problems (an object that only
    gets data on a cold inlet and so never fires), and signal-into-control mismatches.

    Args:
        objects (list): each {"id": str, "type": maxclass, "args": list (optional)}.
        connections (list): each {"from": id, "to": id, "outlet": int=0, "inlet": int=0}.
        problem (str): optional plain-English description of what's going wrong.

    Returns:
        dict: {"problem", "summary", "findings": [{severity, issue, evidence, why, fix}]}.
    """
    return _debug_graph(objects, connections or [], problem or "")


# ── Grounded lessons / curriculum (LLM, drawn only from the reference library) ──
def _grounded_context(question, n_results=12):
    """Retrieve + de-dup + object-inject grounded context for a query.
    Returns {context, sources, best} or None if nothing retrieved."""
    collection, embed_model, bm25, corpus_ids = _get_rag()
    result = _retrieve(
        question,
        collection,
        embed_model,
        bm25,
        corpus_ids,
        n_results=n_results,
        weak_dist=_RAG_WEAK_DIST,
        caution_dist=_RAG_CAUTION_DIST,
    )
    if not result["chunks"]:
        return None
    context = result["context_str"]
    used = sorted({m.get("source", "?") for m in result["metas"]})
    for obj in _objects_in_question(question):
        context = "[AUTHORITATIVE OBJECT REFERENCE]\n" + _format_object_doc(obj) + "\n\n---\n\n" + context
        used.append(obj["name"] + " (object reference)")
    return {"context": context, "sources": sorted(set(used)), "best": result["best_dist"]}


# Lessons should only generate for genuinely covered topics — gate on the BARE
# topic distance (query augmentation would otherwise mask off-topic requests).
_RAG_LESSON_DIST = float(_os.environ.get("RAG_LESSON_DIST", "0.68"))


def _topic_distance(q):
    collection, embed_model, _bm25, _corpus_ids = _get_rag()
    res = collection.query(query_embeddings=[embed_model.encode(q).tolist()],
                           n_results=3, include=["distances"])
    d = res["distances"][0]
    return min(d) if d else 2.0


def _llm_generate(system_prompt, user_message, max_tokens=2200):
    """Single grounded completion. Returns (text, error)."""
    api_key = _get_llm_key()
    if not api_key:
        return None, ("No LLM API key found. Set GENSPARK_API_KEY in the MCP connector "
                      "env (claude_desktop_config.json / ~/.codex/config.toml).")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=_LLM_BASE_URL, timeout=120.0)
        resp = client.chat.completions.create(
            model=_LLM_MODEL, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_message}])
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, f"LLM error: {e}"


_LESSON_SYS = (
    "You are a Max/MSP teacher writing ONE lesson using ONLY the reference material "
    "provided. Never invent object names, inlets, outlets, or arguments — if the "
    "material lacks something, say so plainly instead of fabricating. Output Markdown "
    "with exactly these sections:\n"
    "# <clear lesson title>\n"
    "## Concept — 2-3 plain-English sentences\n"
    "## Prerequisites — what to know first (brief)\n"
    "## Walkthrough — numbered, grounded steps; name exact objects with correct "
    "inlets/outlets from the material\n"
    "## Build This Patch — a small concrete patch: list the objects and how they "
    "connect using real object names (buildable with the build_patch tool)\n"
    "## Exercise — one concrete task for the learner\n"
    "## Sources — the source files you used\n"
    "Plain English first, technical term second. Match the requested level."
)

_CURRICULUM_SYS = (
    "You design a short, ordered Max/MSP learning path toward the user's goal using "
    "ONLY what the reference material supports. Output Markdown: a numbered list of "
    "4-7 lessons; each line is **Title** — one sentence on what it covers and why it "
    "sits here in the sequence. Order prerequisite -> advanced. If part of the goal "
    "isn't covered by the material, say which part is missing rather than inventing "
    "lessons. End with a 'Sources:' line."
)


@mcp.tool()
def generate_lesson(ctx: Context, topic: str, level: str = "beginner") -> dict:
    """Generate ONE structured Max/MSP lesson on a topic, grounded ONLY in the
    reference library (no fabricated objects). Includes Concept, Prerequisites, a
    grounded Walkthrough, a 'Build This Patch' section you can hand to build_patch,
    an Exercise, and Sources.

    Args:
        topic (str): the lesson subject (e.g. "FM synthesis", "step sequencing").
        level (str): "beginner" | "intermediate" | "advanced".

    Returns:
        dict: {"ok": bool, "lesson": markdown, "sources": [...]} or {"ok": False, "message"}.
    """
    if _topic_distance(topic) > _RAG_LESSON_DIST:
        return {"ok": False, "message": f"The reference library doesn't have solid "
                f"material on '{topic}' to build a grounded lesson. Try a more specific "
                "Max/MSP topic."}
    g = _grounded_context(f"{topic} tutorial concept objects how to build")
    if not g:
        return {"ok": False, "message": f"No material retrieved for '{topic}'."}
    user = (f"Reference material:\n\n{g['context']}\n\n---\n\nLesson topic: {topic}\n"
            f"Level: {level}\n(Source files available to cite: {', '.join(g['sources'])})")
    text, err = _llm_generate(_LESSON_SYS, user, max_tokens=2600)
    if err:
        return {"ok": False, "message": err}
    return {"ok": True, "topic": topic, "level": level, "lesson": text, "sources": g["sources"]}


@mcp.tool()
def suggest_curriculum(ctx: Context, goal: str) -> dict:
    """Propose an ordered Max/MSP learning path (4-7 lessons) toward a goal, grounded
    in what the reference library actually covers — flags any part of the goal the
    library doesn't cover rather than inventing lessons. Pair each returned lesson
    title with generate_lesson to produce the full lesson.

    Args:
        goal (str): what the learner wants to achieve (e.g. "build a granular sampler").

    Returns:
        dict: {"ok": bool, "curriculum": markdown, "sources": [...]} or {"ok": False, "message"}.
    """
    if _topic_distance(goal) > _RAG_LESSON_DIST:
        return {"ok": False, "message": f"The reference library doesn't cover '{goal}' "
                "well enough to build a grounded learning path. Try a more specific goal."}
    g = _grounded_context(f"{goal} concepts objects techniques tutorial")
    if not g:
        return {"ok": False, "message": f"No material retrieved for '{goal}'."}
    user = (f"Reference material:\n\n{g['context']}\n\n---\n\nLearner's goal: {goal}\n"
            f"(Source files available to cite: {', '.join(g['sources'])})")
    text, err = _llm_generate(_CURRICULUM_SYS, user, max_tokens=1400)
    if err:
        return {"ok": False, "message": err}
    return {"ok": True, "goal": goal, "curriculum": text, "sources": g["sources"]}


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

    # Grounded pre-flight against the object database. Hard errors (undefined
    # connection ids, missing types) block the build; unknown object types are
    # surfaced as warnings (third-party externals legitimately aren't in the DB).
    _val = _validate_graph(objects, connections)
    if _val["errors"]:
        return {"success": False, "error": "Graph validation failed before building.",
                "validation": _val}

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
        "validation": _val,
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
