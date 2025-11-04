# Optimized RAG System for PDF Q&A

A production-ready RAG (Retrieval-Augmented Generation) system optimized for PDF document question answering.

## 🚀 Key Features

- **Token-Based Chunking** - Precise semantic segmentation (+68% similarity improvement)
- **Smart Deduplication** - Automatic merging of related content from same pages
- **Dynamic Context Management** - Intelligent token control for LLM input
- **Two-Stage Retrieval** - Broad-then-precise for better quality
- **Detailed Metadata** - Course codes, credits, prerequisites, etc.

## 📊 Performance

- **Similarity**: 0.41 → 0.69 (+68%)
- **Chunks**: 59 → 20 (-66%, larger and more complete)
- **Answer Quality**: From "can't find details" → Full course codes and descriptions

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
Copy `.env.example` to `.env` and add your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your-key-here
```

### 3. Run
```bash
# First time - build vector database
python rag.py --rebuild "What are the core courses?"

# Normal usage
python rag.py "Tell me about the capstone project"

# Interactive mode
python rag.py

# Verbose mode
python rag.py --verbose "Your question"
```

## 📖 Example Questions

```bash
python rag.py "What is ADSP 31017 Machine Learning I?"
python rag.py "Tell me about the capstone project requirements"
python rag.py "What are Time Series Analysis course prerequisites?"
```

## 🔧 Configuration

Edit `.env` file:

```env
# Token-based chunking
CHUNK_TOKENS=600
OVERLAP_TOKENS=150

# Retrieval parameters
TOP_K=5
MIN_SIM=0.30

# Generation
MAX_TOKENS=800
```

## 📈 Evaluation

Run RAGAS evaluation:
```bash
python eval_ragas.py
```

Metrics:
- **Faithfulness** - Answer fidelity to context
- **Answer Relevancy** - Relevance to question

## 🛠️ Advanced Options

```bash
# Custom chunk size
python rag.py --chunk_tokens 700 --overlap_tokens 150 "Your question"

# More retrieval results
python rag.py --top_k 8 "Complex question"

# Adjust similarity threshold
python rag.py --min_sim 0.25 "Your question"

# Rebuild database
python rag.py --rebuild "Your question"
```

## 📁 Project Structure

```
project/
├── rag.py              # Main RAG system
├── eval_ragas.py       # RAGAS evaluation
├── .env.example        # Example configuration
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🎯 Technical Highlights

1. **Token-Based Chunking** - Uses `tiktoken` for precise 600-token chunks
2. **Smart Deduplication** - Merges chunks from same page when similarity > 0.3
3. **Two-Stage Retrieval** - Retrieves top_k*2, then deduplicates to top_k
4. **Context Management** - Limits context to 3000 tokens max

## 📊 Performance Metrics

- Response time: ~2-3 seconds
- Similarity range: 0.45 - 0.70
- Accuracy: High (based on retrieved content)

## 📝 Requirements

- Python 3.10+
- OpenAI API key
- Required packages in `requirements.txt`

## 🤝 Contributing

Feel free to open issues or submit PRs!

## 📄 License

MIT
