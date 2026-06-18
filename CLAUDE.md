# CLAUDE.md — maxmsp-mcp-server

MCP server that exposes the Max/MSP RAG corpus to Claude Code and Codex. Fork of the MIT-licensed `tiianhk/MaxMSP-MCP-Server` (see LICENSE; attribution preserved — ADR-0004).

## Setup
```bash
cd ~/Desktop/AI/maxmsp-toolkit/maxmsp-mcp-server
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # venv is NOT committed; recreate after any move
.venv/bin/python3 server.py
```

## Two RAG tools exposed
- `query_maxmsp_docs(question)` — semantic search over the Max/MSP corpus
- `lookup_max_object_reference(object_name)` — inlet/outlet/attribute specs

## Paths (config-driven — no hardcoded absolutes)
- ChromaDB: defaults to `../maxmsp-reference-library/chroma_db` (override: `MAXMSP_CHROMA_PATH`)
- Corpus: defaults to `~/Desktop/AI/maxmsp-toolkit/maxmsp-corpus/licensed/MaxMSP-Corpus` (override: `MAXMSP_CORPUS_DIR`)
- API key: `.env` as `GENSPARK_API_KEY` (gitignored)
- Embedding model: `all-MiniLM-L6-v2` (NEVER change — index was built with it)

## DO NOT
- Change the embedding model
- Edit `chroma_db/` directly
- Change topic taxonomy labels
- Commit `.env`

See `../maxmsp-reference-library/CLAUDE.md` for full RAG rules.
