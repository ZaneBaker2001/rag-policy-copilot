# RAG Policy Copilot

A production-style RAG app for answering questions over policy manuals, contracts, and internal documents with citations, hybrid retrieval, and confidence-based abstention.

## Highlights

- Citation-backed answers grounded in retrieved evidence
- Dense + sparse retrieval with reranking
- Confidence thresholds to reduce hallucinations
- FastAPI `/ask` endpoint for document QA
- FAISS for vector search and SQLite for metadata

## Tech Stack

- Python
- FastAPI
- FAISS
- SQLite
- sentence-transformers
- pytest

## Quick Start 

### Local Environment Setup  

Clone the repository: 
```bash
git clone https://github.com/ZaneBaker2001/rag-policy-copilot.git
cd rag-policy-copilot
```
Activate the virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Install the required packages: 
```bash
pip3 install -r requirements.txt
```
Add the example environment:
```bash
cp .env.example .env
```

### Running the App

Build the index:
```bash
python3 scripts/build_index.py
```
Start the API:
```bash
uvicorn app.main:app --reload
```
Open docs:
```
http://127.0.0.1:8000/docs
```

Sample request: 
```bash 
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -H "x-api-key: dev-admin-key" -d '{"question":"What is the PTO carryover policy?"}'
```

### Add Documents 

Supported file types include:

- .pdf
- .txt
- .md
- .html
- .htm

Two sample .txt files are provided. 

### Environment 

A sample environment file is provided:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DATA_DIR=data/docs
STORAGE_DIR=storage
TOP_K=6
MAX_CONTEXT_CHUNKS=6
CHUNK_SIZE=900
CHUNK_OVERLAP=120
```

This file can be customized.

## Project Structure 

```text
rag-policy-copilot/
├── app/
│   ├── config.py
│   ├── db.py
│   ├── generator.py
│   ├── ingest.py
│   ├── main.py
│   ├── models.py
│   ├── retriever.py
│   └── utils.py
├── data/
│   └── docs/
├── evals/
│   ├── eval_cases.json
│   └── retrieval_eval.py
├── scripts/
│   └── build_index.py
├── storage/
│   ├── id_map.pkl
│   ├── index.faiss
│   └── rag.db
├── tests/
│   ├── test_authz.py
│   ├── test_chunking.py
│   └── test_hybrid_scoring.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Features

- Ingest PDF, TXT, MD, and HTML files from `data/docs/`
- Parse text and preserve section structure when possible
- Chunk documents with paragraph-aware splitting and overlap
- Generate embeddings with `sentence-transformers`
- Store vectors in FAISS
- Store chunk metadata in SQLite
- Retrieve chunks with dense + sparse + rerank scoring
- Apply confidence thresholds and abstain on weak matches
- Ask grounded questions through a FastAPI `/ask` endpoint
- Return citations with source filenames, chunk IDs, and page numbers when available
- Enforce basic API-key auth and chunk-level access filtering
- Run tests with `pytest`
- Run a simple retrieval evaluation script

## Architecture

1. Documents are ingested and chunked
2. Chunks are embedded and stored in FAISS
3. Metadata is stored in SQLite
4. Queries go through dense retrieval, sparse retrieval (BM25), and
reranking
5. Top chunks are passed to the generator
6. Responses include citations and confidence signals

## Purpose

Many LLM apps hallucinate when answering questions over internal documents.
This implementation demonstrates a production-style RAG system that:

- Grounds answers in retrieved evidence
- Abstains when confidence is low
- Surfaces retrieval diagnostics for debugging

It is designed as a reference implementation for building reliable document QA systems.

## API

### GET /health 

Returns service health status 

### POST /ask

Accepts a question and returns:

- An answer
- Retrieved citations
- Applied filters 
- Retrieval diagnostics

## Run Tests

```bash
python3 -m pytest -q
```

## Run Evaluations

To evaluate retrievals:
```bash
python3 -m evals.retrieval_eval
```

## Evaluation Results 

The following results were produced from a sample evaluation run: 

### Retrieval Evaluation Results 

| Section | Metric | Value |
|---|---|---:|
| Overall Metrics | Cases | 38 |
| Overall Metrics | Hit@1 | 86.84% |
| Overall Metrics | Hit@3 | 86.84% |
| Overall Metrics | MRR@5 | 0.8684 |
| Overall Metrics | Confident Rate | 97.37% |
| Overall Metrics | Abstain Rate | 2.63% |

These results indicate that the model operates under a high degree of confidence, abstains when under uncertainty, all while maintaining high accuracy. 

## Limitations 

- Performance not optimized for large-scale datasets
- No distributed indexing
- Limited document parsing for complex PDFs
- Evaluation dataset is small and synthetic




