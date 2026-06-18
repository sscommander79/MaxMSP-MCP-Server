---
name: Build & verify Max patches (live)
description: Use when constructing, editing, or validating a Max/MSP patch in a running Max instance via the maxmsp MCP server — building a patch from scratch, wiring objects, or checking that a patch (built here or emitted by MaxPyLang and opened in Max) matches its intended graph. Triggers: "build a patch", "make a max patch", "wire up", "create objects in max", "verify the patch", "check the patch", "round-trip", "does the patch match".
---

# Build & verify Max patches (live)

The `maxmsp` MCP server can drive a running Max instance (host patch from
`MaxMSP-MCP-Server/MaxMSP_Agent` must be open). Prefer the high-level tools; they
read the patch back and report failures instead of building blind.

## Build from scratch: `build_patch`

Describe the whole graph in one call; the server lays it out (top-to-bottom flow,
parallel nodes side by side, comments in a right-side lane), wires by id, then dumps
the live patch and reports any missing object/connection, self-healing missing cords
once.

```
build_patch(
  objects=[
    {"id":"carrier","type":"cycle~","args":[440]},
    {"id":"amp","type":"*~","args":[0.2]},
    {"id":"out","type":"dac~"}
  ],
  connections=[
    {"from":"carrier","to":"amp"},
    {"from":"amp","to":"out","inlet":0},
    {"from":"amp","to":"out","inlet":1}
  ]
)
```

`id` is the stable scripting name — wire, set, and delete by it. For `message`,
`comment`, and `flonum`, put the text/content in `args`. Read the returned
`connections_missing` / `objects_missing`; if non-empty, fix types/ids or inlet/outlet
indices and rebuild.

## Verify intent: `verify_patch`

Read-only round-trip check. Pass the intended `objects`/`connections`; it dumps the
live patch and reports `*_missing` (intended but absent) and `*_unexpected`
(present but not intended — e.g. objects the user added by hand). Use it to validate
a patch your own builder (MaxPyLang → `.maxpat`) produced, by opening that patch in
Max and diffing against the graph you meant to emit.

## Editing an existing patch

For incremental edits use the unitary tools: `add_max_object`,
`connect_max_objects`, `disconnect_max_objects`, `set_message_text`,
`set_object_attribute`, `set_number`, `send_messages_to_object`,
`send_bang_to_object`, `remove_max_object`. Inspect first with
`get_objects_in_patch` (full dump) or `get_objects_in_selected`.

## House style (follow when building)

- **Decouple value from trigger:** a `number`/`flonum` holds the value; a separate
  `button` fires it. Don't conflate them.
- **Name everything** with a readable `id`; never rely on auto-generated names.
- **Mind hot/cold inlets:** inlet 0 is the hot (triggering) inlet; Max evaluates
  simultaneously-triggered inlets right-to-left. Use `trigger` to enforce order/fan-out.
- **Load a message through its right inlet** rather than `prepend set` where possible.
- **Keep the signal path vertical**, `dac~` at the bottom, no overlapping boxes.
- **Verify by reading back** — trust `build_patch`/`verify_patch` output, not assumption.

## Ground object choices in the reference

Before using an unfamiliar object, look it up via the `maxmsp-reference` skill's
`lookup_max_object_reference` / `get_object_doc` so inlets, outlets, and args are
correct rather than guessed.

## Prerequisite

Max must be running with the host patch open (the Socket.IO bridge on port 5002). If
`build_patch`/`verify_patch` return "Max is not connected", open the host patch and
retry. The RAG lookup tools still work with Max closed.
