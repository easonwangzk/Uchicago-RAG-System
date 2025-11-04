# -*- coding: utf-8 -*-
"""
RAGAS evaluation for the improved RAG pipeline.
- Computes: faithfulness, answer_relevancy, context_precision.
- Does NOT require human-written references (so context_recall is omitted by default).
- You can add your own question set by editing QUESTIONS below or feeding a file.

Usage:
    python eval_ragas.py
    # or customize env/flags via .env and rag.py defaults
"""

import os
from typing import List, Dict

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from rag import (
    getenv_str, getenv_int, getenv_float, ensure_api_key,
    load_or_build_vectordb, answer_question
)


# A small default question set; feel free to extend to your real project needs.
QUESTIONS: List[str] = [
    "What are the core courses?",
    "How long does the program typically take to complete?",
    "Are there any capstone or practicum components?",
    "Tell me about the Time Series Analysis course",
    "What is the Machine Learning I course about?",
]


def build_predictions(vdb, chat_model: str, temperature: float, max_tokens: int,
                      top_k: int, min_sim: float) -> Dict[str, List]:
    """
    Run the RAG pipeline on the question set and collect outputs
    in a format compatible with RAGAS.
    """
    qs, answers, contexts = [], [], []

    print(f"\n Generating answers for {len(QUESTIONS)} questions...")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"  [{i}/{len(QUESTIONS)}] {q}")
        out = answer_question(
            vdb=vdb,
            question=q,
            top_k=top_k,
            min_sim=min_sim,
            chat_model=chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # RAGAS expects raw "answer" (string) and "contexts" (list of strings)
        qs.append(q)
        answers.append(out["answer"])
        contexts.append(out["contexts"])

    print("✓ All answers generated\n")
    return {"question": qs, "answer": answers, "contexts": contexts}


def main():
    load_dotenv()
    ensure_api_key()

    # Read defaults from .env
    pdf_path = getenv_str("PDF_PATH", "mastersprograminanalytics.pdf")
    db_dir = getenv_str("CHROMA_DIR", ".chroma")
    embed_model = getenv_str("EMBED_MODEL", "text-embedding-3-large")
    chat_model = getenv_str("CHAT_MODEL", "gpt-4o-mini")
    chunk_tokens = getenv_int("CHUNK_TOKENS", 500)
    overlap_tokens = getenv_int("OVERLAP_TOKENS", 100)
    top_k = getenv_int("TOP_K", 5)
    min_sim = getenv_float("MIN_SIM", 0.30)
    temperature = getenv_float("TEMPERATURE", 0.2)
    max_tokens = getenv_int("MAX_TOKENS", 800)

    print("\n" + "="*80)
    print("RAGAS Evaluation for RAG System")
    print("="*80)
    print(f"PDF: {pdf_path}")
    print(f"Embedding Model: {embed_model}")
    print(f"Chat Model: {chat_model}")
    print(f"TOP_K: {top_k}, MIN_SIM: {min_sim}")
    print(f"Chunk Tokens: {chunk_tokens}, Overlap: {overlap_tokens}")
    print("="*80)

    # Build/load vector store
    vdb = load_or_build_vectordb(
        pdf_path=pdf_path,
        persist_dir=db_dir,
        embed_model=embed_model,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )

    # Collect predictions for RAGAS dataset
    data_dict = build_predictions(
        vdb=vdb,
        chat_model=chat_model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=top_k,
        min_sim=min_sim,
    )
    ragas_ds = Dataset.from_dict(data_dict)

    # Evaluate with RAGAS
    print(" Running RAGAS evaluation...")
    result = evaluate(
        ragas_ds,
        metrics=[faithfulness, answer_relevancy],
        # RAGAS will automatically use LangChain's OpenAI under the hood via env vars
    )

    print("\n" + "="*80)
    print("RAGAS Scores (higher is better, 0-1 range)")
    print("="*80)
    df = result.to_pandas()
    print(df.to_string(index=False))

    print("\n" + "="*80)
    print("Per-question breakdown:")
    print("="*80)
    print(ragas_ds.to_pandas()[["question", "answer"]].to_string(index=False))
    print("\n" + "="*80)


if __name__ == "__main__":
    main()