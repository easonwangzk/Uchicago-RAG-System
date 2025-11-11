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
from datetime import datetime

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
    qs, answers, contexts, sources, similarities = [], [], [], [], []

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
        sources.append("; ".join(out.get("sources", [])))
        similarities.append(out.get("best_similarity", 0.0))

    print("+ All answers generated\n")

    # Debug: Print sample data
    print(" Sample data check:")
    print(f"  - Questions: {len(qs)}")
    print(f"  - Answers: {len(answers)}")
    print(f"  - Contexts: {len(contexts)}")
    if contexts:
        print(f"  - Sample context type: {type(contexts[0])}")
        print(f"  - Sample context length: {len(contexts[0]) if contexts[0] else 0}")

    return {
        "question": qs,
        "answer": answers,
        "contexts": contexts,
        "sources": sources,
        "similarity": similarities
    }


def main():
    load_dotenv()
    ensure_api_key()

    # Read defaults from .env
    pdf_path = getenv_str("PDF_PATH", "mastersprograminanalytics.pdf")
    html_dir = getenv_str("HTML_DIR", "data")
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
    print("RAGAS Evaluation for RAG System (PDF + HTML)")
    print("="*80)
    print(f"PDF: {pdf_path}")
    print(f"HTML Directory: {html_dir}")
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
        html_dir=html_dir,
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

    # Prepare dataset for RAGAS (only needs question, answer, contexts)
    ragas_input = {
        "question": data_dict["question"],
        "answer": data_dict["answer"],
        "contexts": data_dict["contexts"]
    }
    ragas_ds = Dataset.from_dict(ragas_input)

    # Evaluate with RAGAS
    print(" Running RAGAS evaluation...")

    # Configure LLM and Embeddings for RAGAS
    llm = ChatOpenAI(model=chat_model, temperature=0)
    embeddings = OpenAIEmbeddings(model=embed_model)

    result = evaluate(
        ragas_ds,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n" + "="*80)
    print("RAGAS Scores (higher is better, 0-1 range)")
    print("="*80)
    scores_df = result.to_pandas()
    print(scores_df.to_string(index=False))

    # Debug: Check if scores are NaN
    if scores_df["faithfulness"].isna().all():
        print("\n! WARNING: All faithfulness scores are NaN")
        print("  This may indicate:")
        print("  1. Context format issue (should be list of strings)")
        print("  2. Empty contexts")
        print("  3. RAGAS version compatibility issue")
    if scores_df["answer_relevancy"].isna().all():
        print("\n! WARNING: All answer_relevancy scores are NaN")
        print("  This may indicate embedding or LLM configuration issue")

    print("\n" + "="*80)
    print("Per-question breakdown:")
    print("="*80)
    print(ragas_ds.to_pandas()[["question", "answer"]].to_string(index=False))
    print("\n" + "="*80)

    # Export results to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"ragas_evaluation_{timestamp}.xlsx"

    print(f"\n Exporting results to Excel: {excel_filename}")

    # Create comprehensive DataFrame combining all information
    comprehensive_df = pd.DataFrame({
        "Question": data_dict["question"],
        "Answer": data_dict["answer"],
        "Sources": data_dict["sources"],
        "Best_Similarity": data_dict["similarity"],
        "Faithfulness": scores_df["faithfulness"].fillna(0),  # Replace NaN with 0
        "Answer_Relevancy": scores_df["answer_relevancy"].fillna(0),  # Replace NaN with 0
    })

    # Create a summary statistics DataFrame
    summary_df = pd.DataFrame({
        "Metric": ["Faithfulness", "Answer Relevancy", "Best Similarity"],
        "Mean": [
            scores_df["faithfulness"].mean(),
            scores_df["answer_relevancy"].mean(),
            comprehensive_df["Best_Similarity"].mean()
        ],
        "Min": [
            scores_df["faithfulness"].min(),
            scores_df["answer_relevancy"].min(),
            comprehensive_df["Best_Similarity"].min()
        ],
        "Max": [
            scores_df["faithfulness"].max(),
            scores_df["answer_relevancy"].max(),
            comprehensive_df["Best_Similarity"].max()
        ],
        "Std": [
            scores_df["faithfulness"].std(),
            scores_df["answer_relevancy"].std(),
            comprehensive_df["Best_Similarity"].std()
        ]
    })

    # Create configuration DataFrame
    config_df = pd.DataFrame({
        "Parameter": [
            "PDF Path", "HTML Directory", "Embedding Model", "Chat Model",
            "TOP_K", "MIN_SIM", "Chunk Tokens", "Overlap Tokens",
            "Temperature", "Max Tokens", "Evaluation Date"
        ],
        "Value": [
            pdf_path, html_dir, embed_model, chat_model,
            top_k, min_sim, chunk_tokens, overlap_tokens,
            temperature, max_tokens, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
    })

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Sheet 1: Summary Statistics
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 2: Detailed Results
        comprehensive_df.to_excel(writer, sheet_name='Detailed Results', index=False)

        # Sheet 3: Configuration
        config_df.to_excel(writer, sheet_name='Configuration', index=False)

        # Adjust column widths for better readability
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 100)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"+ Results exported successfully to {excel_filename}")
    print(f"  - Sheet 'Summary': Overall performance metrics")
    print(f"  - Sheet 'Detailed Results': Per-question results with scores")
    print(f"  - Sheet 'Configuration': System configuration parameters")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()