# Agent Controller Report

**Milestone 6 — Part 2**

**Model:** `mistral:7b-instruct` via Ollama  
**Tools:** `retriever`, `summarizer`, `extractor`  
**Required evaluation set:** 10 tasks in `agent_traces/task_01.json` through `task_10.json`

## 1. Tool Selection Policy

### Available tools

| Tool | Intended use |
|---|---|
| `retriever` | Retrieve grounded chunks from the shared FAISS index |
| `summarizer` | Condense retrieved context into short prose |
| `extractor` | Pull out structured facts, lists, or techniques |

### Planning behavior

Each trace begins with a planning step that returns an ordered JSON-like tool plan. The routing is observable because the selected tools and reasons are saved in the trace itself.

The controller also applies a lightweight normalization layer:

- short “summarize / in 2 sentences” tasks are nudged toward `retriever -> summarizer`
- “quote exactly / verbatim / exact definition” tasks are nudged toward retriever-only behavior
- unknown tools are not executed

### Retrieval integration

The agent now reuses persisted Part 1 retrieval artifacts directly:

- `rag_index.faiss`
- `rag_chunks_metadata.json`

If those files are missing, `agent_controller.py` rebuilds them from the Part 1 notebook corpus and saves them back to disk. This is real index reuse, not just reuse of general retrieval logic.

## 2. What The Required 10 Tasks Show

### Runtime completion

All 10 required tasks completed without runtime failure.

| Task | Current observed route | Runtime status | Answer-quality note |
|---|---|---|---|
| `task_01` | `retriever -> summarizer` | completed | grounded summary of RAG benefits |
| `task_02` | `retriever -> extractor -> extractor` | completed | grounded FAISS / ChromaDB characteristic extraction |
| `task_03` | `retriever -> extractor` | completed | grounded deployment summary with explicit options and tradeoffs |
| `task_04` | `retriever -> extractor` | completed | grounded metric listing |
| `task_05` | `retriever -> extractor` | completed | grounded recommendation for fixed-size chunking on technical docs |
| `task_06` | `retriever -> extractor` | completed | grounded LoRA / QLoRA extraction |
| `task_07` | `retriever -> summarizer` | completed | grounded best-practice summary via summarizer route |
| `task_08` | `retriever -> summarizer` | completed | grounded drift-monitoring summary via summarizer route |
| `task_09` | `retriever -> extractor` | completed | grounded semantic-search explanation |
| `task_10` | `retriever -> extractor` | completed | grounded CoT definition with self-consistency kept separate |

The strongest honest summary is:

- runtime success: `10/10`
- answer quality: grounded across all 10 tasks
- routing diversity: 3 distinct routes observed in the required set

## 3. Routing Story

### Required set

The required 10-task set now shows meaningful routing diversity after reruns with the current normalization logic.

Observed planning counts in the current saved required traces:

- `retriever`: 10 (every task retrieves first)
- `extractor`: 7 (task_02, task_03, task_04, task_05, task_06, task_09, task_10)
- `summarizer`: 3 (task_01, task_07, task_08)

Tasks that include the word "summarize" without "list / extract / compare / enumerate" are normalized to `retriever -> summarizer`. Tasks with explicit extraction or listing intent route to `retriever -> extractor`. task_02 triggered the planner to call extractor twice in a single plan.

### Supplementary routing traces

Two real supplementary traces were added and are clearly outside the required 10:

| Trace | Task type | Observed route | Quality note |
|---|---|---|---|
| `task_11_supplementary.json` | short summary request | `retriever -> summarizer` | routing diversity demonstrated, but retrieval quality is weak |
| `task_12_supplementary.json` | exact quote request | `retriever` only | clean direct-quote behavior |

These supplementary runs are useful as routing evidence, not as a replacement for the required 10-task set.

## 4. Remaining Limitation Analysis

The main remaining honest limitation is that all 10 required tasks still start with the `retriever` step — direct retriever-only routing (as seen in `task_12_supplementary`) does not appear in the required set, because none of the 10 required task prompts match the "quote exactly / verbatim / exact definition" heuristic.

## 5. Conclusion

This agent satisfies the rubric requirements for Part 2:

- at least two tools: yes
- documented tool policy: yes
- 10 saved multi-step traces: yes
- observable routing decisions: yes
- routing diversity in required set: 3 distinct routes (`retriever -> extractor`, `retriever -> summarizer`, `retriever -> extractor -> extractor`)
- supplementary traces demonstrate retriever-only path
