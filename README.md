# IDS 568 — Milestone 6: RAG Pipeline and Multi-Tool Agent

This submission contains:

- Part 1: a small local RAG pipeline over 10 MLOps / LLM documents
- Part 2: a multi-tool agent with observable planning and saved task traces

The repository has been cleaned for accuracy and consistency. Claims below are aligned to the current saved artifacts:

- Part 1 raw results: `eval_results_real.json`
- Part 2 required traces: `agent_traces/task_01.json` through `agent_traces/task_10.json`
- Supplementary routing demos: `agent_traces/task_11_supplementary.json`, `agent_traces/task_12_supplementary.json`

![RAG Pipeline Architecture](rag_pipeline_diagram.png)

## What Is Implemented

### Part 1 — RAG Pipeline

- Runnable script path: `rag_pipeline.py`
- Document ingestion from an in-notebook corpus of 10 domain documents
- Fixed-size chunking with 512-character chunks and 100-character overlap
- Dense embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- FAISS `IndexFlatIP` vector search
- Grounded generation via local Ollama using `mistral:7b-instruct`
- Evaluation on 10 handcrafted queries with saved `precision@3`, `recall@3`, explicit groundedness assessment, retrieval latency, generation latency, and answers

### Part 2 — Multi-Tool Agent

- Tools: `retriever`, `summarizer`, `extractor`
- Observable planning step saved in each trace
- 10 required multi-step task traces
- Real supplementary traces showing additional routing patterns

## Shared Retrieval Reuse

The agent now performs real Part 1 index reuse.

- The shared FAISS index is stored in `rag_index.faiss`.
- Chunk metadata aligned to that index is stored in `rag_chunks_metadata.json`.
- `agent_controller.py` loads those persisted artifacts instead of rebuilding a fresh runtime-only index when they are present.
- If the saved artifacts are missing, the agent rebuilds them from the Part 1 notebook corpus and saves them back to disk.

This is a stronger and more truthful form of reuse than the earlier README/report wording.

## Repository Structure

```text
ids568-milestone6-kiran/
├── README.md
├── rag_pipeline.ipynb
├── rag_pipeline.py
├── rag_evaluation_report.md
├── rag_pipeline_diagram.md
├── rag_pipeline_diagram.png
├── rag_evaluation_charts.png
├── eval_results_real.json
├── rag_index.faiss
├── rag_chunks_metadata.json
├── agent_controller.py
├── agent_report.md
├── requirements.txt
├── scripts/
│   └── render_diagram.py
└── agent_traces/
    ├── task_01.json
    ├── task_02.json
    ├── task_03.json
    ├── task_04.json
    ├── task_05.json
    ├── task_06.json
    ├── task_07.json
    ├── task_08.json
    ├── task_09.json
    ├── task_10.json
    ├── task_11_supplementary.json
    └── task_12_supplementary.json
```

## Setup

### 1. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ollama

```bash
ollama serve
ollama pull mistral:7b-instruct
```

### 3. Run the artifacts

```bash
jupyter notebook rag_pipeline.ipynb
python rag_pipeline.py --query "What is RAG?"
python rag_pipeline.py --summary-from-existing eval_results_real.json
python agent_controller.py
```

To recompute the full Part 1 evaluation with real Ollama generation:

```bash
python rag_pipeline.py --evaluate --output-json eval_results_recomputed.json
```

## Part 1 Results

These values are taken directly from `eval_results_real.json`.

| Metric | Value |
|---|---:|
| Hit Rate@3 | 1.00 |
| Hit Rate@1 | 0.90 |
| Precision@3 | 0.700 |
| Recall@3 | 1.000 |
| Mean retrieval latency | 293.4 ms |
| Mean generation latency | 10.2 s |
| Mean end-to-end latency | 10.5 s |

Notable observed issue in the saved Part 1 answers:

- Q3 contains a truncated `ollama pull mistral:7b-instr` command (chunking artifact — the full command spans a chunk boundary).

See [rag_evaluation_report.md](/Users/kiran14/Documents/IDS%20568/IDS-milstone6/ids568-milestone6-kiran/rag_evaluation_report.md) for the full evidence-based writeup.

## Part 2 Results

The required 10 tasks all completed without runtime failure.

- Runtime completion rate: 10/10
- Routing in the required set: `retriever -> extractor` (6/10), `retriever -> summarizer` (3/10 — task_01, task_07, task_08), `retriever` only (task_03), `retriever -> extractor -> extractor` (task_02)
- All 10 traces have fresh timestamps from the current code version

Supplementary traces:

- `task_11_supplementary.json`: real `retriever -> summarizer` path, but retrieval quality is weak, so it is useful mainly as routing evidence
- `task_12_supplementary.json`: real retriever-only quote task with a clean direct quote answer

See [agent_report.md](/Users/kiran14/Documents/IDS%20568/IDS-milstone6/ids568-milestone6-kiran/agent_report.md) for the detailed analysis.

## Notebook Status

`rag_pipeline.ipynb` was cleaned to remove the broken absolute-path install command and to document the Ollama dependency clearly. A matching repo-root CLI path is also provided in `rag_pipeline.py`.

- The install step is now portable: `%pip install -r requirements.txt`
- The notebook source uses repo-relative paths only
- The stale Metal out-of-memory text was removed from the saved evaluation output
- The script path uses the same Part 1 corpus, chunking strategy, embedding model, and shared FAISS artifacts
- The notebook has now been rerun successfully from a clean Jupyter execution path in this repo

Current note:

- `eval_results_real.json` is the authoritative saved Part 1 evaluation artifact because it was regenerated by the notebook rerun

## Submission Notes

This repo now satisfies the structural rubric requirements for both parts, but the submission stays conservative about answer correctness.

- Part 1 is supported by a working pipeline design, saved raw evaluation JSON, a diagram, and a written report
- Part 2 is supported by a real 3-tool controller, 10 required traces, explicit planning output, and an honest routing/limitation discussion
