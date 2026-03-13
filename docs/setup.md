# Setup Guide

## Prerequisites

- Python 3.10+
- Windows PowerShell, Bash, or equivalent shell
- Internet access for package installation

## Clone Repository

```bash
git clone https://github.com/MercuryConnor/marketmind-ai
cd marketmind-ai/financial-ai-assistant
```

## Create and Activate Virtual Environment

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Build Local RAG Index

This step prepares FAISS index files from local documents.

```bash
python -c "from app.rag.index_builder import build_financial_index; build_financial_index()"
```

## Run the API Server

```bash
uvicorn app.main:app --reload
```

## Verify Endpoints

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"running"}
```

### Ask Endpoint

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is P/E ratio?"}'
```

## Run Test Suite

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Troubleshooting

- If FAISS/LlamaIndex retrieval fails, rebuild index and retry.
- If import errors occur, confirm the active interpreter is the project venv.
- If port 8000 is busy, run uvicorn with an alternate port.
