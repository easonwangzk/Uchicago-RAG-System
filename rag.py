# -*- coding: utf-8 -*-
"""
Optimized RAG system with hybrid chunking and improved retrieval.
- Token-based chunking for better semantic preservation
- Larger chunks with overlap for context completeness
- Smart retrieval with re-ranking
- Better prompt engineering
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict
import warnings

from dotenv import load_dotenv
from pypdf import PdfReader
import tiktoken
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --------------------------
# Prompts
# --------------------------
SYSTEM_PROMPT = """You are a knowledgeable assistant answering questions about an academic program based STRICTLY on the provided CONTEXT.

Instructions:
- Use ONLY the information in the CONTEXT to answer
- If the CONTEXT contains the answer, provide it clearly with specific details
- If the CONTEXT doesn't contain the answer, say "I couldn't find that information in the provided context"
- Organize your answer with bullet points or short paragraphs
- Be specific: include course codes, numbers, requirements, etc. when available
- Cite page numbers at the end
"""

USER_TEMPLATE = """CONTEXT (Retrieved from PDF):
{context}

QUESTION: {question}

Please answer based on the context above. Be specific and include all relevant details.
"""


# --------------------------
# Environment helpers
# --------------------------
def getenv_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default

def getenv_int(key: str, default: int) -> int:
    v = os.getenv(key)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default

def getenv_float(key: str, default: float) -> float:
    v = os.getenv(key)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


# --------------------------
# PDF Processing
# --------------------------
def extract_pdf_text_with_metadata(file_path: str) -> List[Dict]:
    """Extract text from PDF with page metadata."""
    reader = PdfReader(file_path)
    pages_data = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages_data.append({
                "text": text,
                "page": i + 1,
                "source": f"{os.path.basename(file_path)}#page={i+1}"
            })

    return pages_data


def chunk_text_by_tokens(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 100
) -> List[str]:
    """
    Chunk text by tokens with overlap to preserve context.
    Using larger chunks than before for better context.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        if chunk_text.strip():
            chunks.append(chunk_text.strip())

        # Move forward with overlap
        start = end - overlap_tokens if end < len(tokens) else end

    return chunks


def create_documents_from_pdf(
    file_path: str,
    chunk_tokens: int = 500,
    overlap_tokens: int = 100
) -> List[Document]:
    """
    Process PDF into Documents with smart chunking.
    """
    pages_data = extract_pdf_text_with_metadata(file_path)
    documents = []

    for page_data in pages_data:
        chunks = chunk_text_by_tokens(
            page_data["text"],
            max_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens
        )

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page": page_data["page"],
                    "source": page_data["source"],
                    "chunk_id": f"p{page_data['page']}_c{i}"
                }
            )
            documents.append(doc)

    return documents


# --------------------------
# Core RAG functions
# --------------------------
def ensure_api_key():
    """Ensure OpenAI API key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Export it or put in .env.")


def load_or_build_vectordb(
    pdf_path: str,
    persist_dir: str,
    embed_model: str,
    chunk_tokens: int,
    overlap_tokens: int,
    force_rebuild: bool = False,
) -> Chroma:
    """
    Load or build Chroma vector database with optimized chunking.
    """
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings(model=embed_model)

    # Try to load existing DB
    if not force_rebuild:
        try:
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            count = vectordb._collection.count()
            if count > 0:
                print(f"✓ Loaded existing vector DB with {count} chunks")
                return vectordb
        except Exception as e:
            print(f"⚠ Could not load existing DB: {e}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Cannot find '{pdf_path}'")

    print(f"📄 Loading PDF: {pdf_path}")
    documents = create_documents_from_pdf(
        pdf_path,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens
    )
    print(f"✓ Created {len(documents)} chunks")

    print(f"🔄 Building vector database with {embed_model}...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"✓ Vector database created")
    return vectordb


def cosine_distance_to_similarity(dist: float) -> float:
    """
    Convert Chroma's cosine distance to similarity [0, 1].
    """
    dist = max(0.0, min(2.0, dist))
    sim = 1.0 - (dist / 2.0)
    return max(0.0, min(1.0, sim))


def retrieve_with_scores(
    vdb: Chroma,
    query: str,
    k: int
) -> List[Tuple[Document, float]]:
    """Retrieve top-k documents with similarity scores."""
    docs_scores = vdb.similarity_search_with_score(query, k=k)
    results = []
    for doc, dist in docs_scores:
        sim = cosine_distance_to_similarity(dist)
        results.append((doc, sim))
    return results


def deduplicate_and_merge_chunks(
    results: List[Tuple[Document, float]]
) -> List[Tuple[Document, float]]:
    """
    Smart deduplication: merge chunks from the same page.
    This helps when multiple chunks from the same page are retrieved.
    """
    page_groups = {}

    for doc, sim in results:
        page = doc.metadata.get("page", 0)
        if page not in page_groups:
            page_groups[page] = []
        page_groups[page].append((doc, sim))

    # For each page, keep the best scoring chunk or merge if multiple
    merged_results = []
    for page, chunks in page_groups.items():
        # Sort by similarity
        chunks.sort(key=lambda x: x[1], reverse=True)

        # If multiple high-quality chunks from same page, merge them
        if len(chunks) > 1 and chunks[1][1] > 0.3:
            # Merge top 2 chunks
            merged_content = f"{chunks[0][0].page_content}\n\n{chunks[1][0].page_content}"
            merged_doc = Document(
                page_content=merged_content,
                metadata=chunks[0][0].metadata
            )
            merged_results.append((merged_doc, chunks[0][1]))
        else:
            merged_results.append(chunks[0])

    # Sort by similarity
    merged_results.sort(key=lambda x: x[1], reverse=True)
    return merged_results


def format_context(results: List[Tuple[Document, float]], max_context_tokens: int = 3000) -> str:
    """Format retrieved chunks into context string with token limit."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    blocks = []
    total_tokens = 0

    for i, (doc, sim) in enumerate(results, 1):
        txt = doc.page_content.strip()
        src = doc.metadata.get("source", "")

        block = f"[Chunk {i}] (Relevance: {sim:.2f}) [Source: {src}]\n{txt}\n"
        block_tokens = len(tokenizer.encode(block))

        if total_tokens + block_tokens > max_context_tokens:
            break

        blocks.append(block)
        total_tokens += block_tokens

    return "\n".join(blocks)


def extract_sources(results: List[Tuple[Document, float]]) -> List[str]:
    """Extract unique sources."""
    sources = []
    seen = set()
    for doc, _ in results:
        src = doc.metadata.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def answer_question(
    vdb: Chroma,
    question: str,
    top_k: int,
    min_sim: float,
    chat_model: str,
    temperature: float,
    max_tokens: int,
    verbose: bool = False,
) -> Dict:
    """
    Full RAG pipeline with improved retrieval and generation.
    """
    llm = ChatOpenAI(model=chat_model, temperature=temperature, max_tokens=max_tokens)

    # Retrieve documents
    results = retrieve_with_scores(vdb, question, k=top_k * 2)  # Get more initially

    if verbose:
        print(f"\n🔍 Initial retrieval: {len(results)} chunks")
        for i, (doc, sim) in enumerate(results[:5], 1):
            src = doc.metadata.get('source', 'N/A')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  [{i}] Sim: {sim:.2f} | {src} | {preview}...")

    # Deduplicate and merge
    results = deduplicate_and_merge_chunks(results)

    # Filter by similarity and keep top_k
    results = [(doc, sim) for doc, sim in results if sim >= min_sim][:top_k]

    if verbose:
        print(f"\n📊 After dedup/filter: {len(results)} chunks")

    if not results:
        return {
            "grounded": False,
            "answer": "I couldn't find relevant information in the PDF.",
            "sources": [],
            "contexts": [],
            "best_similarity": 0.0,
        }

    best_sim = max(sim for _, sim in results)

    # Format context and generate answer
    context_str = format_context(results, max_context_tokens=2500)
    user_prompt = USER_TEMPLATE.format(question=question, context=context_str)

    if verbose:
        print(f"\n💭 Generating answer with {chat_model}...")
        print(f"   Context tokens: ~{len(tiktoken.get_encoding('cl100k_base').encode(context_str))}")

    resp = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ])

    answer = (resp.content or "").strip()
    sources = extract_sources(results)

    # Add sources if not already in answer
    if "Sources:" not in answer and sources:
        answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    return {
        "grounded": True,
        "answer": answer,
        "sources": sources,
        "contexts": [doc.page_content for doc, _ in results],
        "best_similarity": best_sim,
    }


# --------------------------
# CLI
# --------------------------
def main():
    load_dotenv()
    ensure_api_key()

    # Read defaults from .env
    default_pdf = getenv_str("PDF_PATH", "mastersprograminanalytics.pdf")
    default_db = getenv_str("CHROMA_DIR", ".chroma")
    default_embed_model = getenv_str("EMBED_MODEL", "text-embedding-3-large")
    default_chat_model = getenv_str("CHAT_MODEL", "gpt-4o-mini")
    default_chunk_tokens = getenv_int("CHUNK_TOKENS", 500)
    default_overlap_tokens = getenv_int("OVERLAP_TOKENS", 100)
    default_top_k = getenv_int("TOP_K", 5)
    default_min_sim = getenv_float("MIN_SIM", 0.30)
    default_temp = getenv_float("TEMPERATURE", 0.2)
    default_maxtoks = getenv_int("MAX_TOKENS", 800)

    parser = argparse.ArgumentParser(
        description="Optimized RAG system with token-based chunking."
    )
    parser.add_argument("question", type=str, nargs="*",
                       help="Your question (if not provided, will prompt interactively)")
    parser.add_argument("--pdf", default=default_pdf)
    parser.add_argument("--db", default=default_db)
    parser.add_argument("--embed_model", default=default_embed_model)
    parser.add_argument("--chat_model", default=default_chat_model)
    parser.add_argument("--chunk_tokens", type=int, default=default_chunk_tokens)
    parser.add_argument("--overlap_tokens", type=int, default=default_overlap_tokens)
    parser.add_argument("--top_k", type=int, default=default_top_k)
    parser.add_argument("--min_sim", type=float, default=default_min_sim)
    parser.add_argument("--temperature", type=float, default=default_temp)
    parser.add_argument("--max_tokens", type=int, default=default_maxtoks)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild vector database")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Get question
    if args.question:
        q = " ".join(args.question).strip()
    else:
        q = input("\n💬 Enter your question: ").strip()
        if not q:
            print("No question provided. Exiting.")
            sys.exit(0)

    # Load/build vector DB
    vectordb = load_or_build_vectordb(
        pdf_path=args.pdf,
        persist_dir=args.db,
        embed_model=args.embed_model,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        force_rebuild=args.rebuild,
    )

    # Answer question
    out = answer_question(
        vdb=vectordb,
        question=q,
        top_k=args.top_k,
        min_sim=args.min_sim,
        chat_model=args.chat_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    # Display results
    print("\n" + "=" * 80)
    print("Question:", q)
    print("-" * 80)
    print(out["answer"])
    print("=" * 80)
    print(f"Grounded: {out['grounded']} | Best Relevance: {out['best_similarity']:.2f}")
    if out["sources"]:
        print(f"\nTop sources ({len(out['sources'])}):")
        for s in out["sources"][:5]:
            print("  •", s)


if __name__ == "__main__":
    main()
