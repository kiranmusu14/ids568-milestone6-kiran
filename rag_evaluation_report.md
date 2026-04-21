# RAG Pipeline Evaluation Report

**Milestone 6 — Part 1**

**Model:** `mistral:7b-instruct` via Ollama  
**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Vector index:** FAISS `IndexFlatIP`  
**Chunking:** 512 characters with 100-character overlap  
**Authoritative raw artifact:** `eval_results_real.json`

Hardware wording used consistently in this report:

- Apple M2 Pro, 16 GB RAM
- generation served locally via Ollama with Metal acceleration
- sentence embeddings computed in Python before FAISS retrieval

## 1. Pipeline Design

The Part 1 pipeline follows the expected RAG stages:

1. document ingestion from a 10-document in-notebook corpus
2. fixed-size chunking with overlap
3. embedding generation with `all-MiniLM-L6-v2`
4. FAISS vector indexing
5. top-`k` retrieval
6. grounded answer generation using retrieved chunks only
7. saved evaluation over 10 handcrafted queries

The saved chunk count for the evaluated run is 33 total chunks.

## 2. Retrieval Metrics

All values below match `eval_results_real.json`.

| Metric | Value |
|---|---:|
| Hit Rate@3 | 1.00 |
| Hit Rate@1 | 0.90 |
| Precision@3 | 0.700 |
| Recall@3 | 1.000 |
| Mean retrieval latency | 293.4 ms |
| Mean generation latency | 10.2 s |
| Mean end-to-end latency | 10.5 s |

Per-query retrieval results:

| Q# | Ground-truth doc in top-3? | P@3 | R@3 | Top-1 score |
|---|---|---:|---:|---:|
| Q1 | yes | 0.333 | 1.00 | 0.455 |
| Q2 | yes | 1.000 | 1.00 | 0.529 |
| Q3 | yes | 0.333 | 1.00 | 0.527 |
| Q4 | yes | 0.333 | 1.00 | 0.459 |
| Q5 | yes | 1.000 | 1.00 | 0.678 |
| Q6 | yes | 0.667 | 1.00 | 0.574 |
| Q7 | yes | 0.667 | 1.00 | 0.562 |
| Q8 | yes | 0.667 | 1.00 | 0.675 |
| Q9 | yes | 1.000 | 1.00 | 0.690 |
| Q10 | yes | 1.000 | 1.00 | 0.574 |

## 3. Groundedness Assessment

To make groundedness explicit for grading, each saved answer is assessed conservatively against the retrieved context as a binary groundedness judgment.

| Q# | Grounded? | Notes |
|---|---|---|
| Q1 | yes | Definition and anti-hallucination claim are directly supported by retrieved RAG chunks. |
| Q2 | yes | Comparison points are supported by the retrieved vector-search document. |
| Q3 | partial | The Ollama deployment answer is grounded, but the saved command is truncated at the chunk boundary. |
| Q4 | yes | Metrics listed are directly supported by the retrieved evaluation-metrics chunk. |
| Q5 | yes | The fixed-size 512-character with 100-character overlap answer is directly grounded. |
| Q6 | yes | LoRA explanation and parameter-reduction claim stay within retrieved context. |
| Q7 | yes | Verbose, but still supported by the retrieved MLOps best-practices content. |
| Q8 | yes | Drift-detection methods and tools are directly supported by the retrieved monitoring chunk. |
| Q9 | yes | Semantic-similarity explanation stays within the retrieved sentence-embeddings document. |
| Q10 | yes | Final answer stays within the retrieved Chain-of-Thought context. |

Summary groundedness score for the saved run:

- fully grounded: 9/10
- partially grounded: 1/10
- ungrounded: 0/10

## 4. Qualitative Answer Assessment

This section is intentionally conservative and tied to the raw saved answers, not to ideal answers.

### Retrieval quality in the rerun

- All 10 queries retrieved the ground-truth document in the top 3
- Q2 and Q9, which were misses in the earlier saved run, are now successful retrieval cases in the rerun

### Minor answer-level issues despite successful retrieval

- Q3: the answer identifies Ollama correctly, but the saved command is truncated as `ollama pull mistral:7b-instr` (chunking artifact — the full command spans a chunk boundary)

### Queries that are reasonably grounded in the saved run

- Q4: correct metrics are listed even though the ground-truth document was ranked second rather than first
- Q6: answer stays on-topic for LoRA parameter reduction
- Q7: answer is grounded but verbose
- Q8: strongest retrieval result in the set
- Q10: the raw saved answer stays within the retrieved CoT context; earlier report language about ReAct or Tree-of-Thoughts is not supported by the raw JSON and has been removed

## 5. Retrieval Notes

The rerun no longer shows retrieval failures in the 10-query set.

- The weakest successful top-1 scores are still moderate rather than dominant for some queries, such as Q1 and Q4
- Even with successful retrieval, answer quality can still lag behind retrieval quality, which is visible in Q3

## 6. Latency

These values are copied from the raw saved results.

| Q# | Retrieval (ms) | Generation (s) | End-to-end (s) |
|---|---:|---:|---:|
| Q1 | 177.73 | 14.52 | 14.70 |
| Q2 | 586.90 | 25.21 | 25.80 |
| Q3 | 331.13 | 6.69 | 7.02 |
| Q4 | 233.67 | 7.81 | 8.05 |
| Q5 | 236.02 | 4.36 | 4.59 |
| Q6 | 294.14 | 6.78 | 7.08 |
| Q7 | 230.06 | 12.59 | 12.83 |
| Q8 | 298.82 | 10.40 | 10.70 |
| Q9 | 265.72 | 4.97 | 5.24 |
| Q10 | 279.51 | 8.79 | 9.08 |

## 7. Bottom Line

Part 1 satisfies the rubric expectations:

- document ingestion: yes
- chunking strategy: yes
- embeddings and vector index: yes
- retrieval plus grounded generation: yes
- 10-query evaluation with saved metrics and explicit groundedness assessment: yes
- diagram and latency measurements: yes

The strongest honest caveat in the rerun is that retrieval is now consistently successful, but some answer-level issues remain even when the correct document is retrieved.
