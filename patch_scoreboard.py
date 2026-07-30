#!/usr/bin/env python3
"""patch_scoreboard.py — STATIC tier of the Patch Confidence Harness.

The offline, Max-free regression guard for the patch-BUILD half of the toolkit —
the analog of the reference-library's eval_scoreboard.py, but for patch graphs
instead of retrieval. It replays a library of known-good AND known-bad patch
graphs (patch_recipes.json) through the repo's pure validators and asserts each
verdict is unchanged, so a code change that breaks patch validation is caught by
machine rather than in a live Max session.

Validators exercised (both pure, no Max, no LLM, no network):
  - server._validate_graph(objects, connections) -> {ok, errors, warnings}
  - server._debug_graph(objects, connections, problem) -> {..., findings:[{severity, issue, ...}]}

Recipe format (patch_recipes.json): each recipe has {objects, connections, expect}.
'expect' asserts on whichever validators are relevant:
  expect.validate.ok               (bool)  — _validate_graph()["ok"]
  expect.validate.errors_contain   (list)  — each substring must appear in some error
  expect.validate.warnings_contain (list)  — each substring must appear in some warning
  expect.debug.errors_contain      (list)  — each substring must appear in some error-severity finding

Usage:
  python patch_scoreboard.py            # run all recipes, print PASS/FAIL table
  python patch_scoreboard.py --json     # machine-readable summary
Exit code: 0 if every recipe passes, 1 otherwise (gate-able, CI/nightly-safe).

NOTE: the validators live in patch_validators.py (imported by both this harness and
server.py's validate_patch_graph / debug_patch tools) — one copy, no fork, no drift.
That module has no heavy deps, so this harness runs fully offline and fast.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RECIPES_PATH = _HERE / "patch_recipes.json"

# Import the validators from patch_validators.py — the intended single source of
# truth, factored out so this harness stays offline (importing server.py drags in
# the RAG stack: torch/chromadb/sentence-transformers). NOTE: as of this writing
# server.py still holds byte-identical INLINE copies of these validators; the de-dup
# rewire (server.py importing from here) is a flagged follow-up. So this harness
# guards the patch_validators.py copy — parity with server.py's copy is separately
# asserted by test_patch_validators_parity.py.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from patch_validators import _validate_graph, _debug_graph  # noqa: E402
except Exception as e:  # pragma: no cover - environment/import failure
    print(f"FATAL: could not import validators from patch_validators.py: {e}", file=sys.stderr)
    sys.exit(2)


def _missing_substrings(substrs, haystack_texts):
    """Return the substrings NOT found in any of haystack_texts."""
    joined = "\n".join(haystack_texts)
    return [s for s in substrs if s not in joined]


def check_recipe(recipe):
    """Run one recipe. Return (passed: bool, failures: list[str])."""
    objs = recipe.get("objects", [])
    conns = recipe.get("connections", [])
    expect = recipe.get("expect", {})
    failures = []

    ve = expect.get("validate")
    if ve is not None:
        res = _validate_graph(objs, conns)
        if "ok" in ve and res["ok"] != ve["ok"]:
            failures.append(f"validate.ok: expected {ve['ok']}, got {res['ok']} "
                            f"(errors={res['errors']})")
        miss = _missing_substrings(ve.get("errors_contain", []), res["errors"])
        if miss:
            failures.append(f"validate.errors missing {miss}; got errors={res['errors']}")
        miss = _missing_substrings(ve.get("warnings_contain", []), res["warnings"])
        if miss:
            failures.append(f"validate.warnings missing {miss}; got warnings={res['warnings']}")

    de = expect.get("debug")
    if de is not None:
        dbg = _debug_graph(objs, conns, recipe.get("name", ""))
        err_texts = [f["issue"] for f in dbg["findings"] if f.get("severity") == "error"]
        miss = _missing_substrings(de.get("errors_contain", []), err_texts)
        if miss:
            failures.append(f"debug.errors missing {miss}; got error-findings={err_texts}")
        # no_errors gives the GOOD direction teeth: assert the debug layer produced
        # ZERO error-severity findings. Without this, an always-erroring _debug_graph
        # would still pass a recipe that only checks for missing substrings.
        if de.get("no_errors") and err_texts:
            failures.append(f"debug.no_errors: expected zero error findings, got {err_texts}")

    return (not failures), failures


def main():
    ap = argparse.ArgumentParser(description="Static patch-graph regression guard.")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    ap.add_argument("--recipes", default=str(RECIPES_PATH), help="path to patch_recipes.json")
    args = ap.parse_args()

    data = json.loads(Path(args.recipes).read_text())
    recipes = data.get("recipes", [])
    results = []
    for r in recipes:
        passed, failures = check_recipe(r)
        results.append({"name": r.get("name", "?"), "kind": r.get("kind", "?"),
                        "passed": passed, "failures": failures})

    n_pass = sum(1 for r in results if r["passed"])
    n = len(results)

    if args.json:
        print(json.dumps({"passed": n_pass, "total": n, "results": results}, indent=2))
    else:
        width = max((len(r["name"]) for r in results), default=10)
        for r in results:
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"[{mark}] {r['name']:<{width}}  ({r['kind']})")
            for f in r["failures"]:
                print(f"         ↳ {f}")
        print("─" * (width + 24))
        print(f"patch_scoreboard: {n_pass}/{n} recipes pass")

    sys.exit(0 if n_pass == n else 1)


if __name__ == "__main__":
    main()
