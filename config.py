import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME","agentic-ai-gemini-rag")
PINECONE_CLOUD=os.getenv("PINECONE_CLOUD","aws")
PINECONE_REGION=os.getenv("PINECONE_REGION","us-east-1")
PINECONE_NAMESPACE=os.getenv("PINECONE_NAMESPACE","agentic-ai")
PDF_PATH=os.getenv("PDF_PATH","Ebook-Agentic-AI.pdf")
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL","gemini-embedding-001")
EMBEDDING_DIMENSION=int(os.getenv("EMBEDDING_DIMENSION","768"))
LLM_MODEL=os.getenv("LLM_MODEL","gemini-2.5-flash")
CHUNK_SIZE=int(os.getenv("CHUNK_SIZE","800"))
CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP","100"))
TOP_K=int(os.getenv("TOP_K","5"))
MIN_RETRIEVAL_SCORE=float(os.getenv("MIN_RETRIEVAL_SCORE","0.35"))
missing=[n for n,v in {"GEMINI_API_KEY":GEMINI_API_KEY,"PINECONE_API_KEY":PINECONE_API_KEY}.items() if not v]
if missing: raise RuntimeError("Missing required environment variables: "+", ".join(missing))
