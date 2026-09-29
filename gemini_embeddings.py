import math
import time
from typing import List
from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION

BATCH_SIZE = 25          # texts per request
PAUSE_SECONDS = 16       # 25 texts / 16s is about 94 per minute, under the 100 limit


class GeminiEmbeddings(Embeddings):
    def __init__(self, model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIMENSION):
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    @staticmethod
    def _normalize(v):
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def _embed_batch(self, texts, task_type):
        for attempt in range(6):
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.output_dimensionality,
                    ),
                )
                break
            except Exception as e:
                if getattr(e, "code", None) == 429 and attempt < 5:
                    print(f"Rate limited, waiting 30s (retry {attempt + 1}/5)...")
                    time.sleep(30)
                    continue
                raise
        vectors = [list(x.values) for x in response.embeddings]
        if len(vectors) != len(texts):
            raise RuntimeError(f"Gemini returned {len(vectors)} embeddings for {len(texts)} inputs")
        return [self._normalize(v) for v in vectors]

    def embed_documents(self, texts: List[str]):
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            out.extend(self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
            print(f"Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")
            if i + BATCH_SIZE < len(texts):
                time.sleep(PAUSE_SECONDS)
        return out

    def embed_query(self, text: str):
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]