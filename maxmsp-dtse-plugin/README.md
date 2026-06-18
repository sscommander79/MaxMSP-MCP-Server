# maxmsp-dtse — Claude Code plugin

Makes **Claude Code** reference the Max/MSP training tool while you build the Digitakt
Sequencer Expander (DTSE). It bundles:

- the **`maxmsp` MCP server** (RAG over the corpus + live Max patch tools), and
- two **skills** that tell Claude Code *when* to use them: ground every Max/MSP and
  Digitakt answer in the corpus, and build/verify patches with the high-level tools.

## What Claude Code gets

RAG tools (work with Max closed):
`query_maxmsp_docs`, `lookup_max_object_reference`, `get_object_doc`, `list_all_objects`.

Live Max tools (need Max + the host patch open):
`build_patch`, `verify_patch`, `add_max_object`, `connect_max_objects`,
`disconnect_max_objects`, `set_message_text`, `set_object_attribute`, `set_number`,
`send_messages_to_object`, `send_bang_to_object`, `remove_max_object`,
`get_objects_in_patch`, `get_objects_in_selected`, `get_object_attributes`,
`get_avoid_rect_position`.

## Prerequisites

1. The MCP server venv must have its deps. If the server fails to start, run:
   ```
   cd "~/Desktop/AI/maxmsp-mcp-server"
   .venv/bin/python3 -m pip install -r requirements.txt
   ```
   (Sanity check: `.venv/bin/python3 -c "import server; print('ok')"` prints `ok`.)
2. For the **live** patch tools, open Max and load the host patch in
   `MaxMSP-MCP-Server/MaxMSP_Agent/` (the Socket.IO bridge listens on port 5002).
   The RAG tools work without Max.
3. The ChromaDB index lives at `MaxMSP-RAG/chroma_db/` (already built).

## Install into Claude Code

Recommended (one-command marketplace flow). The marketplace lives at the project root
(`maxmsp-mcp-server/.claude-plugin/marketplace.json`). In Claude Code:
```
/plugin marketplace add "~/Desktop/AI/maxmsp-mcp-server"
/plugin install maxmsp-dtse@maxmsp-training-tools
```

Quick per-session test (no marketplace):
```
claude --plugin-dir "~/Desktop/AI/maxmsp-mcp-server/maxmsp-dtse-plugin"
```

Or add to your Claude Code `settings.json`:
```json
{
  "extraKnownMarketplaces": [
    { "source": { "type": "local",
      "path": "/Users/stevencommander/Desktop/AI/maxmsp-mcp-server/maxmsp-dtse-plugin" } }
  ]
}
```

Reload after edits with `/reload-plugins`. Confirm the server with `/mcp` (should list
`maxmsp`) and the skills with `/help`.

## Paths

`mcp-config.json` uses absolute paths to the server's venv python and `server.py`. If
you move the project, update those two paths.

## Notes

- The corpus partitions content with a `visibility` tag (`public` / `private`). Claude
  Code queries the whole index; the partition only matters when you export a public
  build. DRM-derived books and the Digitakt manual are `private`.
- `build_patch` and `verify_patch` reuse the server's existing command protocol and the
  `[v8]` read-back, so no changes to the Max-side agent are required.
