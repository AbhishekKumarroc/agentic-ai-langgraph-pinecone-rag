from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE

pc = Pinecone(api_key=PINECONE_API_KEY)
print("Indexes:", [i.name for i in pc.list_indexes()])
print("Using index:", PINECONE_INDEX_NAME, "| namespace:", PINECONE_NAMESPACE)

idx = pc.Index(PINECONE_INDEX_NAME)
print(idx.describe_index_stats())