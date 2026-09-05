---
name: maxmsp-reference
description: "Consult the Max/MSP reference corpus for object behavior, DSP, or DTSE design questions."
---

# Max/MSP & DTSE Reference

This project (the DT Sequencer Expander) is developed against a local Max/MSP
knowledge base served by the **`maxmsp` MCP server**. That corpus is the source of
truth — it contains the full Cycling '74 object reference, MSP/gen~/Jitter/M4L docs,
DSP theory books, the **Elektron Digitakt II manual (CC/NRPN chart, trig conditions,
parameter locks, micro timing, Euclidean params)**, and authored sequencing-in-Max
references (Euclidean, Markov, arpeggiator, ratcheting, conditional/probability,
MIDI, step-sequencer architecture).

## Rule: ground before you answer

Do **not** answer Max/MSP or Digitakt questions, state an object's inlets/outlets/
arguments, or write `.maxpat`/`gen~`/Max-Python (MaxPyLang) code from memory. First
query the corpus:

- **`query_maxmsp_docs(question)`** — semantic Q&A over the whole library. Use for
  "how do I build X", best practices, sound-design technique, sequencing patterns,
  Digitakt hardware behavior. Returns an expert answer with a patch diagram.
- **`lookup_max_object_reference(object_name)`** — exact reference for one object
  (`cycle~`, `groove~`, `coll`, `zl`, `live.dial`, `jit.matrix`, ...): inlets,
  outlets, arguments, attributes. Use before wiring or arg-setting an object.
- **`get_object_doc(object_name)`** / **`list_all_objects()`** — structured object
  doc / the full object name list, straight from `docs.json`.

When the question is DTSE-specific (which DT control sends which CC, how conditional
trigs behave, how to build an arpeggiator/Markov/Euclidean generator in Max), query
with Digitakt/sequencing terms — that material is in the corpus (tagged private) and
will surface.

## Why

The corpus was built and cleaned specifically so this tool answers MIDI/sequencing/
Digitakt questions accurately. Answering from training-data priors reintroduces the
hallucinated object signatures and wrong CC numbers the corpus exists to prevent.
If a query returns nothing relevant, say so rather than inventing an answer.

## Prerequisite

The `maxmsp` MCP server must be connected. Its RAG tools (`query_maxmsp_docs`,
`lookup_max_object_reference`, `get_object_doc`, `list_all_objects`) work even when
Max itself is closed — only the live patch tools need Max running.
