#!/usr/bin/env python3
"""test_patch_validators_parity.py — guard the single-source-of-truth for the
patch-graph validators.

patch_validators.py is the ONE definition of _validate_graph / _debug_graph /
_resolve_object and the _KNOWN_UI_TYPES / _AUDIO_OUT_OBJS / _NO_WIRE_OK constants.
server.py must IMPORT them from there, not re-inline them (which would reintroduce
the drift risk the extraction removed).

This test asserts, purely statically (parses server.py with `ast`, never imports it,
so no RAG stack), that:
  1. server.py imports all validator names from patch_validators, and
  2. server.py does NOT define any of them inline (no shadowing FunctionDef/Assign).

Fails (exits nonzero) if a future edit re-inlines a validator or drops the import —
i.e. if the single-source guarantee is broken.
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
NAMES = FUNCS + CONSTS + ["flattened_docs"]


def main():
    server_tree = ast.parse(SERVER.read_text())

    # (0) patch_validators actually defines all of them.
    val_tree = ast.parse(VALIDATORS.read_text())
    val_defined = {n.name for n in val_tree.body if isinstance(n, ast.FunctionDef)}
    val_defined |= {t.id for n in val_tree.body if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}
    missing_src = [n for n in NAMES if n not in val_defined]

    # (1) server.py imports all NAMES from patch_validators.
    imported = set()
    for node in server_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "patch_validators":
            imported |= {a.asname or a.name for a in node.names}
    not_imported = [n for n in NAMES if n not in imported]

    # (2) server.py does NOT define any of them inline (would shadow the import).
    inline = []
    for node in server_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            inline.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in NAMES:
                    inline.append(t.id)

    problems = []
    if missing_src:
        problems.append(f"patch_validators.py is missing: {missing_src}")
    if not_imported:
        problems.append(f"server.py does not import from patch_validators: {not_imported}")
    if inline:
        problems.append(f"server.py RE-INLINES (shadows) the shared validators: {sorted(set(inline))}")

    if problems:
        print("PATCH-VALIDATOR SINGLE-SOURCE: BROKEN")
        for p in problems:
            print(f"  {p}")
        print("Fix: server.py must `from patch_validators import ...` and not redefine them.")
        sys.exit(1)
    print(f"patch-validator single-source: OK (server.py imports all {len(NAMES)} "
          "names from patch_validators; no inline shadowing)")
    sys.exit(0)


if __name__ == "__main__":
    main()
