# CLAUDE.md — MaxMSP MCP Server

MCP server that exposes the Max/MSP RAG corpus to Claude Code and Codex.

## Setup
```bash
cd ~/Desktop/AI/MaxMSP-MCP-Server
.venv/bin/python3 server.py
```

## Two RAG tools exposed
- `query_maxmsp_docs(question)` — semantic search over 31,500+ chunks
- `lookup_max_object_reference(object_name)` — inlet/outlet/attribute specs

## Paths
- ChromaDB: `~/Desktop/AI/MaxMSP-RAG/chroma_db/`
- Corpus: `~/Desktop/AI/MaxMSP-Corpus/`
- API key: `.env` as `GENSPARK_API_KEY`
- Embedding model: `all-MiniLM-L6-v2` (NEVER change)

## DO NOT
- Change the embedding model
- Edit chroma_db/ directly
- Change topic taxonomy labels
- Commit .env

See `../MaxMSP-RAG/CLAUDE.md` for full rules.
