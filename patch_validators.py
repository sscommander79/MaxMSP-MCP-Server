"""patch_validators.py — deterministic, Max-free patch-graph validation.

Extracted from server.py so the pure patch-graph logic can be imported WITHOUT
dragging in the RAG stack (retrieval/query → torch/chromadb/sentence-transformers),
which server.py loads at import time. This module is the intended SINGLE SOURCE OF
TRUTH for patch-graph validation.

STATUS (2026-07-29): patch_scoreboard.py imports these. server.py's MCP tools
(validate_patch_graph / debug_patch) still hold BYTE-IDENTICAL inline copies of
_validate_graph/_debug_graph/_resolve_object/etc — a faithful copy was made here at
extraction time. FOLLOW-UP (flagged, needs an MCP-server restart to runtime-verify):
rewire server.py to `from patch_validators import ...` and delete its inline copies,
so there is one copy and no drift. Until then, do NOT edit the logic in only one
place — change both or (preferably) do the rewire first.

Depends only on the standard library + docs.json (the object database). No Max, no
LLM, no network.

Graph format (matches build_patch / verify_patch):
  objects:     each {"id": str, "type": maxclass, "args": list (optional)}
  connections: each {"from": id, "to": id, "outlet": int=0, "inlet": int=0}
"""
from __future__ import annotations

import json
import os

# ── Object database (docs.json) ──────────────────────────────────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_docs_path = os.path.join(_current_dir, "docs.json")
with open(_docs_path, "r") as _f:
    _docs = json.load(_f)
flattened_docs = {}
for _obj_list in _docs.values():
    for _obj in _obj_list:
        flattened_docs[_obj["name"]] = _obj


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
