#!/usr/bin/env python3
"""
agent_controller.py
Multi-Tool Agent with Retrieval Integration
Milestone 6, Part 2 — MLOps Course
"""

import json
import re
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "mistral:7b-instruct"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K = 3
TRACES_DIR = Path("agent_traces")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Document Corpus (same 10 documents as rag_pipeline.ipynb) ─────────────────
DOCUMENTS = {
    "doc_01": {
        "title": "Retrieval-Augmented Generation (RAG)",
        "content": (
            "Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language models "
            "by dynamically retrieving relevant information from an external knowledge base at inference time. "
            "Unlike purely parametric models that rely solely on information encoded during training, RAG systems "
            "combine neural retrieval with language model generation to produce responses grounded in factual, "
            "verifiable sources.\n\n"
            "The RAG pipeline operates in four main stages. First, documents are ingested and split into smaller "
            "chunks using a chunking strategy. Second, each chunk is encoded into a dense embedding vector. Third, "
            "when a user query arrives, it is also embedded and the most relevant chunks are retrieved via vector "
            "similarity search. Fourth, the retrieved chunks are concatenated with the query, and the language model "
            "generates a grounded answer.\n\n"
            "Benefits of RAG include reduced hallucination because responses are grounded in retrieved evidence, "
            "knowledge currency since retrieval corpora can be updated without retraining, source attribution for "
            "verifiable outputs, and cost efficiency since knowledge updates require only corpus changes. RAG is widely "
            "used for question answering over private documents, technical support, and knowledge-intensive NLP tasks.\n\n"
            "Common failure modes include retrieval failures when relevant documents are not retrieved, context window "
            "overflow with too many chunks, and generation errors where the LLM ignores retrieved context. Proper "
            "evaluation must measure both retrieval accuracy and generation faithfulness separately."
        ),
    },
    "doc_02": {
        "title": "Vector Databases and Similarity Search",
        "content": (
            "Vector databases are specialized storage systems designed to index and query high-dimensional embedding "
            "vectors efficiently. They form the backbone of modern RAG systems by enabling fast semantic similarity "
            "search across large document collections.\n\n"
            "FAISS (Facebook AI Similarity Search) is an open-source library for efficient similarity search supporting "
            "both exact and approximate nearest neighbor search. It offers multiple index types: IndexFlatL2 for exact "
            "Euclidean search, IndexFlatIP for inner product (cosine similarity after normalization), IndexIVFFlat for "
            "inverted file approximate search, and HNSW for graph-based approximate search. FAISS is optimal for "
            "in-memory workloads requiring low latency.\n\n"
            "ChromaDB is a purpose-built open-source vector database with persistent storage, metadata filtering, and "
            "a simple Python API. It supports cosine and L2 distance metrics and integrates directly with LangChain "
            "and LlamaIndex. Chroma is ideal when persistence and metadata filtering are required.\n\n"
            "Weaviate provides hybrid search combining vector similarity with BM25 keyword search. Qdrant offers "
            "HNSW-based indexing with payload filtering. For most RAG prototypes, FAISS provides the best "
            "performance-to-setup-complexity ratio, while Chroma offers better developer experience with built-in "
            "persistence."
        ),
    },
    "doc_03": {
        "title": "LLM Deployment and Serving",
        "content": (
            "Deploying large language models for inference requires careful consideration of hardware constraints, "
            "latency requirements, and serving frameworks. Several options exist for local and self-hosted LLM inference.\n\n"
            "Ollama is a lightweight tool for running open-weight models locally. It handles model downloading, "
            "quantization, and provides a REST API. Ollama supports models including Mistral, Llama, Qwen, and Gemma "
            "families. Installation is straightforward: download Ollama, then run ollama pull mistral:7b-instruct. "
            "The Python client allows chat completions via ollama.chat().\n\n"
            "vLLM is a high-throughput serving framework optimized for GPU inference using PagedAttention for efficient "
            "KV cache management. It achieves 10-24x higher throughput than naive implementations and supports continuous "
            "batching. vLLM is ideal for production deployments on GPU servers.\n\n"
            "Hugging Face Transformers provides direct model loading using AutoModelForCausalLM and AutoTokenizer. "
            "Text Generation Inference (TGI) is Hugging Face production serving stack supporting tensor parallelism.\n\n"
            "For 7B models on consumer hardware: Apple Silicon M-series chips can run 7B models at 10-20 tokens per second "
            "using Ollama with Metal acceleration. NVIDIA GPUs with 8GB+ VRAM achieve 50-100 tokens per second. "
            "CPU-only inference is possible but slow at 1-3 tokens per second."
        ),
    },
    "doc_04": {
        "title": "RAG Evaluation Metrics",
        "content": (
            "Evaluating RAG systems requires measuring both retrieval quality and generation faithfulness separately, "
            "as failures in either stage lead to poor overall performance.\n\n"
            "For retrieval evaluation, precision@k measures the fraction of retrieved documents in the top k that are "
            "relevant. Recall@k measures the fraction of all relevant documents retrieved in the top k. Mean Reciprocal "
            "Rank (MRR) measures the average inverse rank of the first relevant document. Normalized Discounted "
            "Cumulative Gain (nDCG) accounts for graded relevance and position.\n\n"
            "For generation evaluation, RAGAS is a framework specifically designed for RAG evaluation with metrics "
            "including faithfulness (is the answer supported by retrieved context?), answer relevance, context precision, "
            "and context recall. Faithfulness is computed by checking if each claim in the generated answer is supported "
            "by the retrieved context.\n\n"
            "For hallucination detection, compare the generated answer against retrieved context to identify claims made "
            "without supporting evidence. BERTScore measures semantic similarity between generated and reference answers. "
            "End-to-end latency must be decomposed into embedding latency, vector search latency, and generation latency "
            "to identify bottlenecks."
        ),
    },
    "doc_05": {
        "title": "Text Chunking Strategies for RAG",
        "content": (
            "Chunking is one of the most important design decisions in a RAG pipeline. The chunk size and overlap "
            "strategy directly affect retrieval quality by determining what information is co-located in a single "
            "retrievable unit.\n\n"
            "Fixed-size character chunking splits text into segments of a specified character count with optional overlap "
            "between adjacent chunks. This approach is simple to implement and provides consistent chunk sizes. The overlap "
            "prevents important information from being split at chunk boundaries. Typical parameters are 256-1024 characters "
            "with 10-20% overlap. Smaller chunks improve retrieval precision but may lack sufficient context for generation.\n\n"
            "Recursive character text splitting attempts to split on semantic boundaries (paragraphs, then sentences, then "
            "words) before falling back to character splits. This preserves natural text structure better than pure "
            "character splitting.\n\n"
            "Semantic chunking uses embedding similarity to identify natural topic boundaries, grouping sentences that are "
            "semantically related. This produces variable-size chunks that better capture complete thoughts but is "
            "computationally expensive.\n\n"
            "For technical documents, fixed-size chunking with 512 characters and 100-character overlap provides a good "
            "balance: chunks are large enough to contain meaningful information, small enough for precise retrieval, and "
            "the overlap handles boundary cases. Token-based chunking using model tokenizers is more precise but requires "
            "the tokenizer as a dependency."
        ),
    },
    "doc_06": {
        "title": "Transfer Learning and Parameter-Efficient Fine-tuning",
        "content": (
            "Transfer learning enables reuse of pretrained model knowledge for new tasks by fine-tuning on task-specific "
            "data. For large language models, full fine-tuning of all parameters is computationally expensive, leading to "
            "parameter-efficient fine-tuning (PEFT) methods.\n\n"
            "Low-Rank Adaptation (LoRA) adds trainable low-rank decomposition matrices to frozen transformer weight "
            "matrices. If a weight matrix W has dimension d by d, LoRA represents the update as delta W equals A times B "
            "where A has dimensions d by r and B has dimensions r by d with r much less than d. This reduces trainable "
            "parameters by 99 percent compared to full fine-tuning "
            "while achieving comparable performance. LoRA is typically inserted into attention query and value projections.\n\n"
            "QLoRA combines LoRA with 4-bit quantization of the base model weights. The base model is quantized to NF4 "
            "format, reducing memory by 75 percent, while LoRA adapters remain in full precision. This enables fine-tuning 7B "
            "models on single 24GB GPUs.\n\n"
            "Instruction tuning trains models on instruction-response pairs to follow natural language instructions. "
            "Reinforcement Learning from Human Feedback (RLHF) aligns models with human preferences using a reward model. "
            "Direct Preference Optimization (DPO) achieves similar alignment without explicit reward modeling."
        ),
    },
    "doc_07": {
        "title": "MLOps Best Practices",
        "content": (
            "MLOps (Machine Learning Operations) encompasses practices for deploying, monitoring, and maintaining machine "
            "learning models in production. It bridges the gap between ML development and software engineering.\n\n"
            "Experiment tracking with tools like MLflow, Weights & Biases, or DVC enables reproducibility by logging "
            "hyperparameters, metrics, artifacts, and code versions for every training run. Reproducibility requires pinning "
            "random seeds, documenting hardware configurations, and version-controlling datasets.\n\n"
            "Model versioning using model registries enables promotion of models through development, staging, and production "
            "stages with associated metadata. Each model version should include training data lineage, evaluation metrics, "
            "and deployment constraints.\n\n"
            "CI/CD for ML automates testing, training, and deployment pipelines. Data validation tests verify schema and "
            "statistical properties. Unit tests cover preprocessing and inference code. Feature stores centralize feature "
            "computation. Container-based deployment with Docker ensures environment consistency. Kubernetes enables scalable "
            "serving with automatic scaling and rolling deployments. Pipeline orchestration with Airflow or Prefect manages "
            "complex workflows with dependency tracking and failure recovery."
        ),
    },
    "doc_08": {
        "title": "Monitoring ML Models in Production",
        "content": (
            "Production ML systems require continuous monitoring to detect performance degradation caused by data drift, "
            "concept drift, or infrastructure issues. Without monitoring, model quality can silently degrade.\n\n"
            "Data drift occurs when the statistical distribution of input features shifts from the training distribution. "
            "Detection methods include population stability index (PSI), Kolmogorov-Smirnov test for continuous features, "
            "chi-squared test for categorical features, and Maximum Mean Discrepancy (MMD). Tools like Evidently AI, "
            "WhyLogs, and Great Expectations automate drift detection.\n\n"
            "Concept drift occurs when the relationship between input features and target labels changes even if input "
            "distribution remains stable. This requires monitoring prediction accuracy and, when ground truth is available, "
            "model performance metrics over time.\n\n"
            "For LLM monitoring, track input and output token distributions, response latency, error rates, and user "
            "feedback signals. Hallucination monitoring can use embedding similarity between retrieved context and generated "
            "responses. Toxicity classifiers can flag harmful outputs. LangSmith and Arize Phoenix provide LLM-specific "
            "observability. Alerting should use Prometheus with Grafana dashboards. Shadow mode deployment runs new model "
            "versions alongside production without serving predictions to users, enabling safe comparison."
        ),
    },
    "doc_09": {
        "title": "Sentence Embeddings and Semantic Search",
        "content": (
            "Sentence embeddings are dense vector representations that capture the semantic meaning of text in a continuous "
            "high-dimensional space. Semantically similar sentences are mapped to nearby vectors, enabling efficient "
            "semantic similarity computation.\n\n"
            "The sentence-transformers library (SBERT) provides pretrained models optimized for sentence-level semantic "
            "similarity. Models like all-MiniLM-L6-v2 (384 dimensions, 22M parameters) offer an excellent balance of speed "
            "and quality for English text. all-mpnet-base-v2 (768 dimensions) provides higher quality.\n\n"
            "SBERT trains using a siamese network structure with contrastive learning objectives. The model encodes two "
            "sentences and optimizes cosine similarity to be high for semantically related pairs and low for unrelated "
            "pairs. This produces embeddings where cosine similarity directly reflects semantic similarity.\n\n"
            "For semantic search, query and document embeddings are precomputed. At query time, the query embedding is "
            "computed and cosine similarity is measured against all document embeddings. FAISS enables sub-linear search "
            "with approximate methods for large corpora. Domain-specific embedding models (medical, legal, scientific) "
            "often outperform general-purpose models on specialized corpora."
        ),
    },
    "doc_10": {
        "title": "Prompt Engineering Techniques",
        "content": (
            "Prompt engineering is the practice of designing input prompts to elicit desired behaviors from large language "
            "models. Effective prompting can significantly improve model performance without any parameter updates.\n\n"
            "Zero-shot prompting directly asks the model to perform a task without examples. Few-shot prompting includes "
            "2-8 input-output examples in the prompt to demonstrate the desired format and behavior.\n\n"
            "Chain-of-thought (CoT) prompting encourages models to generate intermediate reasoning steps before producing "
            "a final answer. Adding phrases like 'Let's think step by step' significantly improves performance on multi-step "
            "reasoning tasks, arithmetic, and logical inference. CoT is most effective for models above 10B parameters.\n\n"
            "Self-consistency sampling runs the model multiple times with temperature > 0 and takes the majority vote among "
            "generated answers. This improves CoT accuracy by 10-20% at the cost of multiple inference calls.\n\n"
            "System prompts establish the model's persona, constraints, and behavior. Role prompting can improve response "
            "quality for specialized domains. For RAG prompts, explicitly instructing the model to answer only from provided "
            "context reduces hallucination. Including a 'if the context does not contain this information, say so' instruction "
            "improves groundedness."
        ),
    },
}


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Fixed-size character chunking with overlap."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def build_chunk_list(documents: dict) -> List[Dict[str, Any]]:
    """Convert documents to a flat list of chunks with metadata."""
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, doc in documents.items():
        chunks = chunk_text(doc["content"])
        for i, text in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "doc_id": doc_id,
                    "doc_title": doc["title"],
                    "text": text,
                }
            )
    return all_chunks


# ── Index building ─────────────────────────────────────────────────────────────
def build_index(
    all_chunks: List[Dict[str, Any]], embed_model: SentenceTransformer
) -> faiss.IndexFlatIP:
    """Encode chunks and build FAISS IndexFlatIP."""
    texts = [c["text"] for c in all_chunks]
    logger.info(f"Encoding {len(texts)} chunks …")
    embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=32)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dimension}")
    return index


# ── Tool: Retriever ───────────────────────────────────────────────────────────
class RetrieverTool:
    """Retrieves relevant document chunks from the FAISS index."""

    name = "retriever"
    description = (
        "Retrieves relevant information from the knowledge base. "
        "Use when the task requires factual grounding or external knowledge."
    )

    def __init__(
        self,
        all_chunks: List[Dict],
        index: faiss.IndexFlatIP,
        embed_model: SentenceTransformer,
    ) -> None:
        self.all_chunks = all_chunks
        self.index = index
        self.embed_model = embed_model

    def __call__(self, query: str, k: int = TOP_K) -> Dict[str, Any]:
        t0 = time.time()
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = self.all_chunks[int(idx)]
                results.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "doc_title": chunk["doc_title"],
                        "text": chunk["text"],
                        "relevance_score": float(score),
                    }
                )
        latency_ms = (time.time() - t0) * 1000
        return {
            "tool": "retriever",
            "query": query,
            "results": results,
            "latency_ms": round(latency_ms, 2),
            "num_results": len(results),
        }


# ── Tool: Summarizer ──────────────────────────────────────────────────────────
class SummarizerTool:
    """Condenses retrieved text into concise key points."""

    name = "summarizer"
    description = (
        "Summarizes long text into concise key points. "
        "Use when retrieved context is lengthy or the task asks for an overview or summary."
    )

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def __call__(self, text: str, max_words: int = 150) -> Dict[str, Any]:
        t0 = time.time()
        prompt = (
            f"Summarize the following text in at most {max_words} words. "
            "Focus only on the most important points:\n\n"
            f"{text}\n\nSummary:"
        )
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 512},
        )
        summary = response["message"]["content"].strip()
        latency_ms = (time.time() - t0) * 1000
        return {
            "tool": "summarizer",
            "original_word_count": len(text.split()),
            "summary": summary,
            "summary_word_count": len(summary.split()),
            "latency_ms": round(latency_ms, 2),
        }


# ── Tool: Extractor ───────────────────────────────────────────────────────────
class ExtractorTool:
    """Extracts structured facts or lists from text."""

    name = "extractor"
    description = (
        "Extracts key facts, lists, or structured data from text. "
        "Use when the task asks to list, extract, compare, or enumerate specific items."
    )

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def __call__(
        self, text: str, extraction_type: str = "key_facts"
    ) -> Dict[str, Any]:
        t0 = time.time()
        if extraction_type == "key_facts":
            prompt = (
                "Extract the 5 most important facts from this text as a numbered list:\n\n"
                f"{text}\n\nKey Facts:"
            )
        elif extraction_type == "techniques":
            prompt = (
                "List all techniques, methods, or approaches mentioned in this text:\n\n"
                f"{text}\n\nTechniques:"
            )
        else:
            prompt = f"Extract the main points from this text:\n\n{text}\n\nMain Points:"

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 512},
        )
        extracted = response["message"]["content"].strip()
        latency_ms = (time.time() - t0) * 1000
        return {
            "tool": "extractor",
            "extraction_type": extraction_type,
            "extracted": extracted,
            "latency_ms": round(latency_ms, 2),
        }


# ── Agent Controller ───────────────────────────────────────────────────────────
class AgentController:
    """
    Multi-tool agent that intelligently selects between retrieval,
    summarization, and extraction to solve multi-step tasks.

    Decision flow:
      1. PLAN  — LLM decides which tools to use and in what order.
      2. EXECUTE — tools run sequentially, accumulating context.
      3. GENERATE — LLM synthesises accumulated context into a final answer.
      4. TRACE — every step is logged and saved to agent_traces/.
    """

    def __init__(
        self,
        retriever: RetrieverTool,
        summarizer: SummarizerTool,
        extractor: ExtractorTool,
        model_name: str = MODEL_NAME,
        verbose: bool = True,
    ) -> None:
        self.retriever = retriever
        self.summarizer = summarizer
        self.extractor = extractor
        self.model_name = model_name
        self.verbose = verbose
        self.tools: Dict[str, Any] = {
            "retriever": retriever,
            "summarizer": summarizer,
            "extractor": extractor,
        }
        TRACES_DIR.mkdir(exist_ok=True)

    # ── Planning ──────────────────────────────────────────────────────────────
    def _plan(self, task: str) -> List[Dict[str, str]]:
        """Use LLM to decide which tools to invoke and in what order."""
        tool_list = "\n".join(
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        )
        prompt = (
            "You are an intelligent agent that solves tasks using tools.\n\n"
            f"Available tools:\n{tool_list}\n\n"
            f"Task: {task}\n\n"
            "Decide which tools to use and in what order. "
            "Return a JSON array of steps (tool name + reason). "
            "Use at most 3 steps. Example:\n"
            '[{"tool": "retriever", "reason": "need context"}, '
            '{"tool": "summarizer", "reason": "condense output"}]\n\n'
            "Return ONLY the JSON array, no other text."
        )
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["message"]["content"].strip()
        try:
            plan = json.loads(content)
            if not isinstance(plan, list):
                raise ValueError("Plan is not a list")
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\[.*?\]", content, re.DOTALL)
            if match:
                try:
                    plan = json.loads(match.group())
                except json.JSONDecodeError:
                    plan = [
                        {"tool": "retriever", "reason": "retrieve relevant context"},
                        {"tool": "summarizer", "reason": "summarise for final answer"},
                    ]
            else:
                plan = [
                    {"tool": "retriever", "reason": "retrieve relevant context"},
                    {"tool": "summarizer", "reason": "summarise for final answer"},
                ]
        # Enforce max 3 steps (excluding final generation)
        return plan[:3]

    # ── Final answer generation ────────────────────────────────────────────────
    def _generate_final_answer(self, task: str, context: str) -> str:
        prompt = (
            "You are a helpful assistant. Answer the following task using ONLY the information provided.\n"
            "If the information is insufficient, say so explicitly.\n\n"
            f"Task: {task}\n\n"
            f"Information gathered:\n{context}\n\n"
            "Final Answer:"
        )
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 512},
        )
        return response["message"]["content"].strip()

    # ── Main execution loop ────────────────────────────────────────────────────
    def run_task(self, task: str, task_id: str) -> Dict[str, Any]:
        """Execute a multi-step task with full trace logging."""
        trace: Dict[str, Any] = {
            "task_id": task_id,
            "task": task,
            "model": self.model_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "steps": [],
            "final_answer": None,
            "total_latency_ms": 0.0,
            "success": False,
            "failure_reason": None,
        }
        t_total = time.time()

        if self.verbose:
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"Task {task_id}: {task}")
            logger.info("=" * 70)

        try:
            # ── Step 0: Planning ───────────────────────────────────────────────
            t0 = time.time()
            plan = self._plan(task)
            plan_latency = (time.time() - t0) * 1000

            trace["steps"].append(
                {
                    "step": 0,
                    "action": "planning",
                    "decision": plan,
                    "latency_ms": round(plan_latency, 2),
                    "reasoning": "LLM selected tools based on task requirements",
                }
            )
            if self.verbose:
                logger.info(f"Plan ({plan_latency:.0f} ms): {plan}")

            # ── Execute tool steps ─────────────────────────────────────────────
            accumulated_context: List[str] = []
            retrieved_texts: List[str] = []

            for i, step in enumerate(plan):
                tool_name = step.get("tool", "retriever")
                reason = step.get("reason", "")

                if self.verbose:
                    logger.info(f"Step {i+1}: {tool_name} — {reason}")

                step_trace: Dict[str, Any] = {
                    "step": i + 1,
                    "tool": tool_name,
                    "reason": reason,
                    "input_preview": None,
                    "output": None,
                    "latency_ms": None,
                }

                if tool_name == "retriever":
                    result = self.retriever(task)
                    step_trace["input_preview"] = task[:120]
                    step_trace["output"] = result
                    step_trace["latency_ms"] = result["latency_ms"]
                    for r in result["results"]:
                        retrieved_texts.append(r["text"])
                        accumulated_context.append(
                            f"[Source: {r['doc_title']} | score={r['relevance_score']:.3f}]\n{r['text']}"
                        )
                    if self.verbose:
                        for r in result["results"]:
                            logger.info(
                                f"  Retrieved: {r['doc_title']} (score={r['relevance_score']:.3f})"
                            )

                elif tool_name == "summarizer":
                    text_in = "\n\n".join(retrieved_texts) if retrieved_texts else task
                    result = self.summarizer(text_in)
                    step_trace["input_preview"] = text_in[:120] + "…"
                    step_trace["output"] = result
                    step_trace["latency_ms"] = result["latency_ms"]
                    accumulated_context.append(f"[Summary]\n{result['summary']}")
                    if self.verbose:
                        logger.info(f"  Summary: {result['summary'][:100]}…")

                elif tool_name == "extractor":
                    text_in = "\n\n".join(retrieved_texts) if retrieved_texts else task
                    extraction_type = step.get("extraction_type", "key_facts")
                    result = self.extractor(text_in, extraction_type)
                    step_trace["input_preview"] = text_in[:120] + "…"
                    step_trace["output"] = result
                    step_trace["latency_ms"] = result["latency_ms"]
                    accumulated_context.append(f"[Extracted Facts]\n{result['extracted']}")
                    if self.verbose:
                        logger.info(f"  Extracted: {result['extracted'][:100]}…")

                else:
                    logger.warning(f"Unknown tool '{tool_name}', skipping.")

                trace["steps"].append(step_trace)

            # ── Final answer generation ────────────────────────────────────────
            context_text = "\n\n---\n\n".join(accumulated_context)
            t0 = time.time()
            final_answer = self._generate_final_answer(task, context_text)
            gen_latency = (time.time() - t0) * 1000

            trace["steps"].append(
                {
                    "step": len(plan) + 1,
                    "action": "final_answer_generation",
                    "latency_ms": round(gen_latency, 2),
                    "output_preview": (
                        final_answer[:200] + "…" if len(final_answer) > 200 else final_answer
                    ),
                }
            )
            trace["final_answer"] = final_answer
            trace["success"] = True

            if self.verbose:
                logger.info(f"\nFinal Answer (first 300 chars):\n{final_answer[:300]}")

        except Exception as exc:
            trace["failure_reason"] = str(exc)
            logger.error(f"Task {task_id} failed: {exc}")

        trace["total_latency_ms"] = round((time.time() - t_total) * 1000, 2)

        # Save trace
        trace_file = TRACES_DIR / f"{task_id}.json"
        with open(trace_file, "w") as fh:
            json.dump(trace, fh, indent=2)

        if self.verbose:
            logger.info(f"Trace saved → {trace_file} | total={trace['total_latency_ms']:.0f} ms")

        return trace


# ── Evaluation tasks ───────────────────────────────────────────────────────────
EVAL_TASKS = [
    {
        "id": "task_01",
        "task": (
            "Find information about RAG systems and summarize the key benefits of "
            "using retrieval-augmented generation over standard LLM generation."
        ),
    },
    {
        "id": "task_02",
        "task": (
            "What vector databases are available for similarity search? "
            "Extract the main characteristics and use cases for FAISS and Chroma."
        ),
    },
    {
        "id": "task_03",
        "task": (
            "How do I deploy a 7B language model locally? "
            "Summarize the available deployment options and their latency tradeoffs."
        ),
    },
    {
        "id": "task_04",
        "task": (
            "What evaluation metrics should I use to assess my RAG pipeline? "
            "Find and list the key metrics with explanations."
        ),
    },
    {
        "id": "task_05",
        "task": (
            "Compare different text chunking strategies for RAG. "
            "Which strategy works best for technical documents and why?"
        ),
    },
    {
        "id": "task_06",
        "task": (
            "I want to fine-tune an LLM efficiently on limited hardware. "
            "Find information about parameter-efficient methods and extract the key techniques."
        ),
    },
    {
        "id": "task_07",
        "task": (
            "What are the best practices for reproducible ML experiments? "
            "Summarize the key recommendations for experiment tracking and versioning."
        ),
    },
    {
        "id": "task_08",
        "task": (
            "How do I detect data drift in a production ML model? "
            "Find relevant information and summarize the detection approaches and tools."
        ),
    },
    {
        "id": "task_09",
        "task": (
            "Explain how sentence embeddings capture semantic similarity and why "
            "they are useful for building semantic search systems."
        ),
    },
    {
        "id": "task_10",
        "task": (
            "What is chain-of-thought prompting and in what scenarios does it "
            "significantly improve model performance? List the main prompting techniques."
        ),
    },
]


# ── Component builder (shared with rag_pipeline.ipynb logic) ──────────────────
def build_rag_components(
    model_name: str = MODEL_NAME,
) -> Tuple[RetrieverTool, SummarizerTool, ExtractorTool]:
    """Build and return all three agent tools."""
    logger.info("Loading embedding model …")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    all_chunks = build_chunk_list(DOCUMENTS)
    logger.info(f"Total chunks: {len(all_chunks)}")

    index = build_index(all_chunks, embed_model)

    retriever = RetrieverTool(all_chunks, index, embed_model)
    summarizer = SummarizerTool(model_name)
    extractor = ExtractorTool(model_name)
    return retriever, summarizer, extractor


# ── CLI entry point ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Tool Agent Controller — Milestone 6, Part 2"
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Ollama model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Run a single task by ID (e.g. task_01). Omit to run all 10.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    # Verify Ollama connection
    try:
        ollama.list()
        logger.info(f"Ollama reachable. Using model: {args.model}")
    except Exception as exc:
        logger.error(
            f"Cannot connect to Ollama: {exc}\n"
            "Please run 'ollama serve' and 'ollama pull mistral:7b-instruct' first."
        )
        raise SystemExit(1)

    # Build components
    retriever, summarizer, extractor = build_rag_components(args.model)
    agent = AgentController(retriever, summarizer, extractor, args.model, verbose)

    # Select tasks
    tasks = EVAL_TASKS
    if args.task_id:
        tasks = [t for t in EVAL_TASKS if t["id"] == args.task_id]
        if not tasks:
            logger.error(f"Task '{args.task_id}' not found.")
            raise SystemExit(1)

    # Run tasks
    results = []
    for t in tasks:
        trace = agent.run_task(task=t["task"], task_id=t["id"])
        results.append(
            {
                "task_id": t["id"],
                "success": trace["success"],
                "total_latency_ms": trace["total_latency_ms"],
            }
        )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    successes = sum(1 for r in results if r["success"])
    logger.info(f"Tasks: {len(results)} | Successes: {successes} | Failures: {len(results)-successes}")
    for r in results:
        status = "✓" if r["success"] else "✗"
        logger.info(f"  {status} {r['task_id']}  ({r['total_latency_ms']:.0f} ms)")
    logger.info(f"Traces saved to: {TRACES_DIR.resolve()}/")


if __name__ == "__main__":
    main()
