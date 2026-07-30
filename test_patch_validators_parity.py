#!/usr/bin/env python3
"""test_patch_validators_parity.py — guard against drift between patch_validators.py
and server.py's inline copies of the same validators.

patch_validators.py is the intended single source of truth, but until the de-dup
rewire lands, server.py still holds inline copies of _validate_graph / _debug_graph /
_resolve_object and the _KNOWN_UI_TYPES / _AUDIO_OUT_OBJS / _NO_WIRE_OK constants.
This test asserts the two copies are STRUCTURALLY IDENTICAL (AST-equal, ignoring
comments/formatting) so an edit to one that isn't mirrored in the other fails CI.

Purely static: parses both files with `ast`, never imports server.py (which would
drag in the RAG stack). Fast, offline, gate-able (exits nonzero on any drift).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SERVER = _HERE / "server.py"
VALIDATORS = _HERE / "patch_validators.py"

FUNCS = ["_resolve_object", "_validate_graph", "_debug_graph"]
CONSTS = ["_KNOWN_UI_TYPES", "_AUDIO_OUT_OBJS", "_NO_WIRE_OK"]


def _defs(tree):
    """Map name -> AST-normalized source for top-level functions and the RHS of
    top-level assignments of interest."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            out[node.name] = ast.dump(node, annotate_fields=False)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in CONSTS:
                    out[t.id] = ast.dump(node.value, annotate_fields=False)
    return out


def main():
    server = _defs(ast.parse(SERVER.read_text()))
    validators = _defs(ast.parse(VALIDATORS.read_text()))

    drift = []
    for name in FUNCS + CONSTS:
        if name not in server:
            drift.append(f"{name}: MISSING from server.py")
        elif name not in validators:
            drift.append(f"{name}: MISSING from patch_validators.py")
        elif server[name] != validators[name]:
            drift.append(f"{name}: DRIFT — server.py and patch_validators.py differ")

    if drift:
        print("PATCH-VALIDATOR PARITY: DRIFT DETECTED")
        for d in drift:
            print(f"  {d}")
        print("Mirror the change in both files, or do the de-dup rewire "
              "(server.py imports from patch_validators.py).")
        sys.exit(1)
    print(f"patch-validator parity: OK ({len(FUNCS)} functions + {len(CONSTS)} "
          "constants identical in server.py and patch_validators.py)")
    sys.exit(0)


if __name__ == "__main__":
    main()
