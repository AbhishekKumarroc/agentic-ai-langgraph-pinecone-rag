# Agentic AI Gemini RAG Chatbot

A closed-book Retrieval-Augmented Generation (RAG) chatbot that answers questions **only** from the eBook *Agentic AI for Executives* (`Ebook-Agentic-AI.pdf`). If the answer is not in the document, it refuses instead of guessing.

**Architecture:**
PDF → Gemini Embeddings → Pinecone → LangGraph → Gemini 2.5 Flash → Streamlit

| Component | Technology |
|---|---|
| Embeddings | `gemini-embedding-001` (768 dimensions) |
| Vector database | Pinecone (serverless, cosine metric) |
| Orchestration | LangGraph |
| LLM | `gemini-2.5-flash` |
| UI | Streamlit |

---

## How it works

The question passes through a four-step LangGraph pipeline:

```
Question
   │
   ▼
retrieve   → embeds the question and fetches the top-K chunks from Pinecone
   │
   ▼
generate   → Gemini answers using ONLY the retrieved context
   │
   ▼
grade      → a second Gemini call checks that every claim is supported by
   │         the context and returns a confidence score
   ▼
finalize   → returns the answer, or the refusal message if not grounded
```

Refusal message: `I do not have enough information based on the provided document.`

---

## Project structure

```
.
├── app.py                # Streamlit UI
├── rag_pipeline.py       # LangGraph pipeline (retrieve → generate → grade → finalize)
├── ingest.py             # Loads the PDF, chunks it, embeds it, uploads to Pinecone
├── gemini_embeddings.py  # LangChain-compatible Gemini embeddings wrapper (with rate limiting)
├── config.py             # Reads settings from .env
├── requirements.txt      # Python dependencies
├── .env                  # Your API keys (never commit this)
├── .gitignore
└── Ebook-Agentic-AI.pdf  # Source document (place in project root)
```

---

## Prerequisites

- Python 3.10 or newer
- A [Google Gemini API key](https://aistudio.google.com/apikey)
- A [Pinecone API key](https://app.pinecone.io)
- `Ebook-Agentic-AI.pdf` in the project root

---

## Setup

**1. Create and activate a virtual environment (Windows PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```powershell
python -m pip install -r requirements.txt
```

**3. Create your `.env` file** in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key

PINECONE_INDEX_NAME=agentic-ai-gemini-rag
PINECONE_NAMESPACE=agentic-ai
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=768
LLM_MODEL=gemini-2.5-flash

PDF_PATH=Ebook-Agentic-AI.pdf
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=5
```

Only the two API keys are required. Everything else falls back to the defaults shown above.

> **Important:** Use a **new** Pinecone index for this project. Older MiniLM indexes are 384-dimensional; the Gemini embeddings here are 768-dimensional, and the two are not compatible.

---

## Ingest the document

Run this **once** (or again whenever the PDF or chunk settings change):

```powershell
python ingest.py
```

You should see the page count, the chunk count, then progress lines such as `Embedded 25/134 chunks`, ending with `Ingestion complete.`

Notes:

- The free Gemini tier allows about 100 embedding requests per minute, so ingestion is throttled and takes roughly 2 minutes.
- Chunk IDs are deterministic, so re-running ingestion overwrites existing vectors instead of duplicating them.
- Warnings about `fontTools` and `langchain-community` are harmless.

Verify the upload with the Pinecone dashboard, or with a short script that prints `index.describe_index_stats()`. `total_vector_count` should match the chunk count and the `agentic-ai` namespace should appear.

---

## Run the app

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501 and ask a question.

Using `python -m streamlit` instead of plain `streamlit` avoids the "command not found" error when the virtual environment's Scripts folder is not on your PATH.

---

## Output format

Every answer is returned as JSON (shown in the app under **Required JSON Response**):

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI is ... (Page 8).",
  "retrieved_context_chunks": ["chunk text 1", "chunk text 2"],
  "confidence_score": 95.0
}
```

| Key | Description |
|---|---|
| `query` | The user's question |
| `final_answer` | Grounded answer, or the refusal message |
| `retrieved_context_chunks` | The text chunks retrieved from Pinecone |
| `confidence_score` | Grounding confidence, 0 to 100 (0 when the answer is refused) |

---

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | required | Gemini API access |
| `PINECONE_API_KEY` | required | Pinecone access |
| `PINECONE_INDEX_NAME` | `agentic-ai-gemini-rag` | Index to create and query |
| `PINECONE_NAMESPACE` | `agentic-ai` | Namespace inside the index |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Where a new index is created |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `EMBEDDING_DIMENSION` | `768` | Must match the index dimension |
| `LLM_MODEL` | `gemini-2.5-flash` | Answer and grading model |
| `PDF_PATH` | `Ebook-Agentic-AI.pdf` | Document to ingest |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Text splitting |
| `TOP_K` | `5` | Chunks retrieved per question |

---

## Example questions

**Answerable from the book**

- What is Agentic AI?
- What is the difference between RPA and Agentic AI?
- What is the BDI model?
- What are the structural layers in a multi-agent system?
- What are the four readiness levels for adopting Agentic AI?
- What was the impact of the Retail Copilot use case?

**Should be refused**

- Who is the CEO of Google?
- What is the capital of France?
- How do I train a neural network in PyTorch?

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `streamlit` is not recognized | Activate the virtual environment, or use `python -m streamlit run app.py`. |
| Answer is always the refusal message, and **Retrieved Context Chunks** is empty | The Pinecone index or namespace has no vectors. Run `python ingest.py`, and confirm `PINECONE_INDEX_NAME` and `PINECONE_NAMESPACE` match between ingestion and the app. |
| `429 RESOURCE_EXHAUSTED` during ingestion | Free-tier embedding rate limit. Wait a minute and re-run; the throttled `gemini_embeddings.py` handles this. Enabling billing on the Gemini project removes the limit. |
| `Index ... has dimension X; Gemini index needs 768` | You are pointing at an old index. Set a new `PINECONE_INDEX_NAME`. |
| `Missing required environment variables` | `.env` is missing or a key is empty. It must sit in the project root. |
| `PDF not found` | Put `Ebook-Agentic-AI.pdf` in the project root, or fix `PDF_PATH`. |
| Streamlit shows old behavior after a code change | Stop it with Ctrl+C and start it again. |

---

## Security

- Never commit `.env`. It is listed in `.gitignore`.
- If a key is ever exposed, rotate it in the Gemini or Pinecone console.

---

## Limitations

- Answers depend on the quality of PDF text extraction; text inside images (for example the diagrams) is not captured.
- The confidence score comes from an LLM grading step, so treat it as an indicator rather than a calibrated probability.
- The bot answers about one document only and will refuse anything outside it.
