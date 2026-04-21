#!/usr/bin/env python3
"""
rag_pipeline.py
Reproducible CLI for the Milestone 6 Part 1 RAG pipeline.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import ollama

from agent_controller import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL_NAME,
    INDEX_PATH,
    MODEL_NAME,
    TOP_K,
    build_chunk_list,
    build_index,
    load_embedding_model,
    load_index_artifacts,
    load_part1_documents,
    save_index_artifacts,
)

EVAL_RESULTS_PATH = Path("eval_results_real.json")


def load_query_set(path: Path = EVAL_RESULTS_PATH) -> List[Dict[str, Any]]:
    """Load the saved 10-query evaluation spec from the authoritative JSON artifact."""
    with path.open("r") as fh:
        rows = json.load(fh)
    return [
        {
            "q_num": row["q_num"],
            "query": row["query"],
            "query_type": row.get("query_type", "unknown"),
            "relevant_docs": row["relevant_docs"],
        }
        for row in rows
    ]


def get_rag_components(rebuild_index: bool = False) -> Dict[str, Any]:
    """Load or rebuild the shared Part 1 retrieval components."""
    embed_model = load_embedding_model()
    documents = load_part1_documents()

    loaded = None if rebuild_index else load_index_artifacts()
    if loaded is None:
        all_chunks = build_chunk_list(documents)
        index = build_index(all_chunks, embed_model)
        save_index_artifacts(index, all_chunks)
    else:
        index, all_chunks = loaded

    return {
        "documents": documents,
        "all_chunks": all_chunks,
        "index": index,
        "embed_model": embed_model,
    }


def retrieve(
    query: str,
    all_chunks: List[Dict[str, Any]],
    index: faiss.Index,
    embed_model: Any,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """Run dense retrieval for a single query."""
    started = time.perf_counter()
    query_vec = embed_model.encode(query, show_progress_bar=False, convert_to_numpy=True)
    query_vec = query_vec.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)
    retrieval_ms = (time.perf_counter() - started) * 1000.0

    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = all_chunks[idx]
        retrieved.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "doc_title": chunk["doc_title"],
                "text": chunk["text"],
                "score": float(score),
            }
        )
    return {"retrieved": retrieved, "retrieval_ms": retrieval_ms}


def build_generation_prompt(query: str, retrieved: List[Dict[str, Any]]) -> str:
    """Create a grounded-only generation prompt."""
    context = "\n\n".join(
        [
            f"[{item['doc_id']} | {item['doc_title']}]\n{item['text']}"
            for item in retrieved
        ]
    )
    return (
        "Answer the question using only the retrieved context below.\n"
        "If the context does not contain the answer, say so directly.\n"
        "Do not expand acronyms unless the expansion appears in the context.\n"
        "Do not introduce outside facts.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:"
    )


def generate_answer(query: str, retrieved: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Generate a grounded answer through Ollama."""
    prompt = build_generation_prompt(query, retrieved)
    started = time.perf_counter()
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={"temperature": 0, "num_predict": 512},
    )
    generation_s = time.perf_counter() - started
    return {"answer": response["response"].strip(), "generation_s": generation_s}


def score_retrieval(retrieved: List[Dict[str, Any]], relevant_docs: List[str]) -> Dict[str, Any]:
    """Compute precision@k and recall@k for one query."""
    retrieved_docs = [item["doc_id"] for item in retrieved]
    relevant = set(relevant_docs)
    hits = sum(1 for doc_id in retrieved_docs if doc_id in relevant)
    precision_at_k = hits / len(retrieved_docs) if retrieved_docs else 0.0
    recall_hits = len(set(retrieved_docs) & relevant)
    recall_at_k = recall_hits / len(relevant) if relevant else 0.0
    return {
        "retrieved_docs": retrieved_docs,
        "retrieved_titles": [item["doc_title"] for item in retrieved],
        "top1_score": retrieved[0]["score"] if retrieved else 0.0,
        "precision_at_3": precision_at_k,
        "recall_at_3": recall_at_k,
    }


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate the key retrieval and latency metrics."""
    hit_rate_at_3 = statistics.mean(1.0 if row["recall_at_3"] > 0 else 0.0 for row in rows)
    hit_rate_at_1 = statistics.mean(
        1.0 if row["retrieved_docs"] and row["retrieved_docs"][0] in row["relevant_docs"] else 0.0
        for row in rows
    )
    return {
        "hit_rate_at_3": hit_rate_at_3,
        "hit_rate_at_1": hit_rate_at_1,
        "precision_at_3": statistics.mean(row["precision_at_3"] for row in rows),
        "recall_at_3": statistics.mean(row["recall_at_3"] for row in rows),
        "mean_retrieval_ms": statistics.mean(row["retrieval_ms"] for row in rows),
        "mean_generation_s": statistics.mean(row["generation_s"] for row in rows),
        "mean_end_to_end_s": statistics.mean(row["end_to_end_s"] for row in rows),
    }


def print_summary(rows: List[Dict[str, Any]]) -> None:
    """Print a compact metric summary for quick verification."""
    metrics = summarize_results(rows)
    print("RAG evaluation summary")
    print(f"- queries: {len(rows)}")
    print(f"- hit rate@3: {metrics['hit_rate_at_3']:.3f}")
    print(f"- hit rate@1: {metrics['hit_rate_at_1']:.3f}")
    print(f"- precision@3: {metrics['precision_at_3']:.3f}")
    print(f"- recall@3: {metrics['recall_at_3']:.3f}")
    print(f"- mean retrieval ms: {metrics['mean_retrieval_ms']:.1f}")
    print(f"- mean generation s: {metrics['mean_generation_s']:.1f}")
    print(f"- mean end-to-end s: {metrics['mean_end_to_end_s']:.1f}")


def run_demo(args: argparse.Namespace) -> None:
    """Run a single-query retrieval or full RAG demo."""
    components = get_rag_components(rebuild_index=args.rebuild_index)
    result = retrieve(
        args.query,
        components["all_chunks"],
        components["index"],
        components["embed_model"],
        top_k=args.top_k,
    )

    print(f"Query: {args.query}")
    print(f"Retrieval latency: {result['retrieval_ms']:.1f} ms")
    print("Top chunks:")
    for idx, item in enumerate(result["retrieved"], start=1):
        snippet = item["text"].replace("\n", " ")[:180]
        print(
            f"{idx}. {item['doc_id']} | {item['doc_title']} | score={item['score']:.3f}\n"
            f"   {snippet}..."
        )

    if args.skip_generation:
        return

    answer = generate_answer(args.query, result["retrieved"], args.model)
    print(f"\nGeneration latency: {answer['generation_s']:.1f} s")
    print("Answer:")
    print(answer["answer"])


def run_evaluation(args: argparse.Namespace) -> None:
    """Recompute the saved 10-query evaluation set with real retrieval and generation."""
    ollama.list()
    components = get_rag_components(rebuild_index=args.rebuild_index)
    query_set = load_query_set(args.query_set)
    rows: List[Dict[str, Any]] = []

    for item in query_set:
        started = time.perf_counter()
        retrieval = retrieve(
            item["query"],
            components["all_chunks"],
            components["index"],
            components["embed_model"],
            top_k=args.top_k,
        )
        answer = generate_answer(item["query"], retrieval["retrieved"], args.model)
        scored = score_retrieval(retrieval["retrieved"], item["relevant_docs"])
        end_to_end_s = time.perf_counter() - started

        rows.append(
            {
                "q_num": item["q_num"],
                "query": item["query"],
                "query_type": item["query_type"],
                "relevant_docs": item["relevant_docs"],
                **scored,
                "retrieval_ms": retrieval["retrieval_ms"],
                "generation_s": answer["generation_s"],
                "end_to_end_s": end_to_end_s,
                "answer": answer["answer"],
            }
        )
        print(f"Finished Q{item['q_num']} in {end_to_end_s:.1f}s")

    with open(args.output_json, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved recomputed evaluation to {args.output_json}")
    print_summary(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 6 Part 1 RAG pipeline CLI"
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Ollama model name for grounded generation (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--query",
        help="Run a single retrieval/generation demo query",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of retrieved chunks to return (default: {TOP_K})",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Run retrieval only for the demo query",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Recompute the 10-query evaluation with real Ollama generation",
    )
    parser.add_argument(
        "--query-set",
        type=Path,
        default=EVAL_RESULTS_PATH,
        help="JSON file containing the authoritative 10-query evaluation spec",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("eval_results_recomputed.json"),
        help="Output path for a recomputed evaluation run",
    )
    parser.add_argument(
        "--summary-from-existing",
        type=Path,
        default=None,
        help="Print a metric summary from an existing evaluation JSON and exit",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help=(
            "Ignore the persisted FAISS artifacts and rebuild the shared index from "
            "the Part 1 corpus."
        ),
    )
    args = parser.parse_args()

    if args.summary_from_existing is not None:
        with open(args.summary_from_existing, "r") as fh:
            rows = json.load(fh)
        print_summary(rows)
        return

    if args.evaluate:
        run_evaluation(args)
        return

    if not args.query:
        parser.error("Provide --query for a demo run, or use --evaluate / --summary-from-existing.")

    run_demo(args)


if __name__ == "__main__":
    main()
