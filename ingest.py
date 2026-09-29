from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from config import *
from gemini_embeddings import GeminiEmbeddings
import time

def ensure_index():
    pc=Pinecone(api_key=PINECONE_API_KEY)
    names=set()
    for x in pc.list_indexes():
        name=x.get("name") if isinstance(x,dict) else getattr(x,"name",None)
        if name: names.add(name)
    if PINECONE_INDEX_NAME not in names:
        print(f"Creating Pinecone index {PINECONE_INDEX_NAME} ({EMBEDDING_DIMENSION} dimensions)...")
        pc.create_index(name=PINECONE_INDEX_NAME,dimension=EMBEDDING_DIMENSION,metric="cosine",spec=ServerlessSpec(cloud=PINECONE_CLOUD,region=PINECONE_REGION))
        while True:
            d=pc.describe_index(PINECONE_INDEX_NAME)
            status=getattr(d,"status",None)
            ready=status.get("ready",False) if isinstance(status,dict) else getattr(status,"ready",False)
            if ready: break
            time.sleep(2)
    else:
        d=pc.describe_index(PINECONE_INDEX_NAME)
        dim=getattr(d,"dimension",None)
        if dim is not None and dim != EMBEDDING_DIMENSION:
            raise RuntimeError(f"Index {PINECONE_INDEX_NAME} has dimension {dim}; Gemini index needs {EMBEDDING_DIMENSION}. Use a new index name.")

def ingest_pdf():
    pdf=Path(PDF_PATH)
    if not pdf.exists(): raise FileNotFoundError(f"PDF not found: {pdf.resolve()}")
    print("Loading PDF...")
    docs=PyPDFLoader(str(pdf)).load()
    print(f"Loaded {len(docs)} pages.")
    splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP,separators=["\n\n","\n",". "," ",""])
    docs=splitter.split_documents(docs)
    print(f"Created {len(docs)} chunks.")
    for i,doc in enumerate(docs):
        doc.metadata["source"]=pdf.name
        doc.metadata["page"]=int(doc.metadata.get("page",0))+1
        doc.metadata["chunk_id"]=i
    ensure_index()
    print("Initializing Gemini embeddings...")
    store=PineconeVectorStore(index_name=PINECONE_INDEX_NAME,embedding=GeminiEmbeddings(),pinecone_api_key=PINECONE_API_KEY,namespace=PINECONE_NAMESPACE)
    ids=[f"{pdf.stem}-page-{d.metadata['page']}-chunk-{d.metadata['chunk_id']}" for d in docs]
    print("Uploading vectors to Pinecone...")
    store.add_documents(docs,ids=ids)
    print("Ingestion complete.")
    print(f"Index={PINECONE_INDEX_NAME}, dimension={EMBEDDING_DIMENSION}, chunks={len(docs)}")

if __name__=="__main__": ingest_pdf()
