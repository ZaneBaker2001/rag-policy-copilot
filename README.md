# RAG Policy Copilot

A RAG application for answering questions over policy manuals, contracts, and internal documents using hybrid retrieval, grounded citations, and confidence-based abstention.

This implementation demonstrates how to build a more reliable document QA system by combining dense and sparse retrieval, reranking, citation-backed answers, and abstention when supporting evidence is weak. 

## Features

### Ingestion
- PDF, TXT, MD, HTML support
- Structure-aware parsing

### Retrieval
- Dense (embeddings) + sparse (BM25)
- Reranking

### Generation
- Citation-grounded responses
- Confidence-based abstention

### Infrastructure
- FastAPI endpoint
- API-key auth
- SQLite + FAISS storage

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

Docker setup:

```bash
docker build -t rag-policy-copilot .
docker run --rm -p 8000:8000 rag-policy-copilot
```

Sample request: 

```bash 
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -H "x-api-key: dev-admin-key" \
-d '{"question":"What is the PTO carryover policy?"}'
```

Sample response:


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
├── Dockerfile 
├── README.md
└── requirements.txt
```

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

## Evaluations

To evaluate model latency:

```bash
python3 -m evals.latency_eval
```

The below plot visualizes the model's latency.

![Model Latency](./evals/output/latency_by_case.png) 

Latency is mostly stable, with total response time for most cases landing around 160–220 ms. The biggest contributor is usually generation latency, while retrieval latency is smaller but occasionally spikes. There are a few slow outliers, especially case 1 and also cases 12, 23, 32, and 34, where total latency rises noticeably above the baseline. Case 36 is unusual in the opposite direction, with near-zero generation latency, likely indicating an early exit, cached path, or abstention/fallback behavior. Performance is generally consistent and fast, but a few outliers suggest intermittent retrieval or generation bottlenecks.

To evaluate model hallucinations:

```bash
python3 -m evals.hallucination_eval
```

The below plot visualizes the model's hallucinations. 

![Model Hallucinations](./evals/output/hallucination_by_case.png) 

The model is usually fairly grounded, with most cases clustering around an unsupported-sentence ratio of about 0.10 to 0.15. That suggests most answers contain only a small amount of potentially unsupported content. There are a few clear problem cases, though: cases 19, 20, and 38 are elevated, 35 and 37 are moderately high, and case 36 is the major outlier at 1.0, meaning that response appears completely unsupported by the retrieved context. Hallucination risk is generally low-to-moderate, but there are a handful of failure cases that deserve targeted inspection.

To evaluate 

To evaluate retrievals:
```bash
python3 -m evals.retrieval_eval
```

The following results were produced from a sample run: 

| Metric         |   Value |
|----------------|--------:|
| Cases          |      38 |
| Hit@1          |  86.84% |
| Hit@3          |  86.84% |
| MRR@5          |  86.84% |
| Confident Rate |  97.37% |
| Abstain Rate   |   2.63% |

These results indicate that the model operates with great confidence and refrains from providing incorrect answers, which aligns with the
intended behavior.



