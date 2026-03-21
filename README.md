# RAG Policy Copilot

A Retrieval-Augmented Generation (RAG) app for answering questions over policy manuals, contracts, and internal documents with citations.

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

## Quick Start 

Setup the enviornment: 
```bash
python3 -m venv .venv
source .venv/bin/activate   
pip3 install -r requirements.txt
cp .env.example .env
```
## Add Documents 

Supported file types include:

- .pdf
- .txt
- .md
- .html
- .htm 

Sample .txt files are provided. 

## Running the App

Build the index:
```bash
python3 scripts/build_index.py
```
Run the API:
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

## Project Structure 

```text
rag-policy-copilot/
├── app/
│   ├── __init__.py
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

## Example 

Below is a sample QA query that sheds light on the app's behavior under uncertainty. 

**Question:**
What is the PTO carryover policy?

**Answer:**
I do not have enough reliable information in the indexed documents to answer that confidently.

This example confirms that the system does not attempt to provide an answer whenever its retrieval confidence is low.


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

## Run Evals

```bash
python3 -m evals.retrieval_eval
```

## API Request Results 

The following results are from a sample API request:

| Field | Value |
|---|---|
| Question | `What is the PTO carryover policy?` |
| Answer | `I do not have enough reliable information in the indexed documents to answer that confidently. Reason: low_dense_score.` |
| Abstained | `true` |
| Reason | `low_dense_score` |
| Top Score | `0.8` |
| Second Score | `0.2` |
| Margin | `0.6000000000000001` |
| Candidate Count | `2` |
| Used Filters | `{}` |

### Returned Citations

| Rank | Source | Chunk ID | Score | Dense Score | Sparse Score | Rerank Score | Page | Section | Notes |
|---|---|---|---:|---:|---:|---:|---|---|---|
| 1 | `hr_policy.txt` | `hr_policy.txt::chunk_0` | `0.8` | `0.16160598397254944` | `0.14433756729740646` | `-10.549288749694824` | `null` | `null` | Mentions PTO allocation, but not carryover policy |
| 2 | `vendor_agreement.txt` | `vendor_agreement.txt::chunk_0` | `0.2` | `0.14235614240169525` | `0.25607375986579195` | `-11.024574279785156` | `null` | `null` | Retrieved as a lower-relevance candidate |

## Retrieval Evaluation Results

The following results are from a sample evaluation run:

| Question | Confident | Top Source 1 | Top Source 2 |
|---|---|---|---|
| `How many paid time off days do full-time employees receive?` | `true` | `hr_policy.txt` | `vendor_agreement.txt` |
| `How many days per week may employees work remotely with manager approval?` | `true` | `hr_policy.txt` | `vendor_agreement.txt` |
| `How many business days are invoices due within?` | `true` | `vendor_agreement.txt` | `hr_policy.txt` |
| `How long must shared proprietary information remain confidential after termination?` | `true` | `vendor_agreement.txt` | `hr_policy.txt` |

### Summary Metrics

| Metric | Value |
|---|---|
| Hit@1 | `100.00%` |
| Hit@3 | `100.00%` |

Note: These results are based on a small synthetic evaluation set for demonstration purposes.