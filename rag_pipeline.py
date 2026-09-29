import json
import re
from typing import List, TypedDict

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    TOP_K,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)

from gemini_embeddings import GeminiEmbeddings


REFUSAL = "I do not have enough information based on the provided document."


class RAGState(TypedDict, total=False):
    question: str
    context: List[str]
    evidence: List[dict]
    answer: str
    confidence_score: float
    grounded: bool


# ---------------------------------------------------------
# Gemini Embeddings
# ---------------------------------------------------------

embeddings = GeminiEmbeddings()


# ---------------------------------------------------------
# Pinecone
# ---------------------------------------------------------

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
    namespace=PINECONE_NAMESPACE,
)


# ---------------------------------------------------------
# Gemini LLM
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0,
    google_api_key=GEMINI_API_KEY,
)


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def response_to_text(response) -> str:

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                if "text" in item:
                    parts.append(str(item["text"]))

            else:
                parts.append(str(item))

        return "".join(parts)

    return str(content)


def parse_json(text: str) -> dict:

    text = text.strip()

    # Remove markdown JSON fences
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)

    except Exception:

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match:

            try:
                return json.loads(match.group(0))

            except Exception:
                pass

    return {}


# =========================================================
# NODE 1 — RETRIEVAL
# =========================================================

def retrieve_node(state: RAGState):

    question = state["question"]

    print("\n================ RETRIEVAL ================")
    print("Question:", question)

    results = vectorstore.similarity_search_with_score(
        question,
        k=TOP_K
    )

    evidence = []
    context = []

    for index, (doc, score) in enumerate(results):

        page = doc.metadata.get(
            "page",
            "?"
        )

        source = doc.metadata.get(
            "source",
            "Ebook-Agentic-AI.pdf"
        )

        score = float(score)

        print(
            f"Chunk {index + 1}: "
            f"page={page}, "
            f"score={score:.4f}"
        )

        evidence.append(
            {
                "source": source,
                "page": page,
                "retrieval_score": round(score, 4),
                "text": doc.page_content,
            }
        )

        context.append(
            f"""
[Source: {source}
Page: {page}
Retrieval Score: {score:.4f}]

{doc.page_content}
"""
        )

    print("Retrieved chunks:", len(context))
    print("===========================================\n")

    return {
        "context": context,
        "evidence": evidence,
    }


# =========================================================
# NODE 2 — GENERATION
# =========================================================

def generate_node(state: RAGState):

    context = "\n\n".join(
        state.get("context", [])
    )

    question = state["question"]

    if not context:

        return {
            "answer": REFUSAL
        }

    prompt = f"""
You are a strict document-grounded RAG assistant.

You are answering questions about ONE document:

Agentic AI eBook.

IMPORTANT RULES:

1. You may ONLY use information explicitly contained
   in the Context below.

2. You MUST NOT use your own knowledge.

3. You MUST NOT use information from the internet.

4. You MUST NOT make assumptions.

5. If the Context does not contain enough information
   to answer the question, respond EXACTLY with:

{REFUSAL}

6. If the Context contains the answer, provide a clear,
   concise answer.

7. When possible, mention the page number from the context.

Question:

{question}


Context:

{context}
"""

    response = llm.invoke(prompt)

    answer = response_to_text(response).strip()

    if not answer:
        answer = REFUSAL

    return {
        "answer": answer
    }


# =========================================================
# NODE 3 — HALLUCINATION / GROUNDING CHECK
# =========================================================

def grade_node(state: RAGState):

    answer = state.get(
        "answer",
        REFUSAL
    )

    context = "\n\n".join(
        state.get("context", [])
    )

    question = state["question"]

    # If generation already refused,
    # there is nothing to grade.

    if answer == REFUSAL:

        return {
            "grounded": False,
            "confidence_score": 0.0
        }

    prompt = f"""
You are a hallucination detector for a closed-book RAG system.

Question:

{question}


Generated Answer:

{answer}


Retrieved Context:

{context}


Determine whether EVERY important claim in the generated
answer is supported by the retrieved context.

Do NOT use outside knowledge.

Return ONLY valid JSON:

{{
    "grounded": true,
    "confidence": 0.95
}}

Rules:

- grounded=true only when the answer is supported by
  the retrieved context.

- grounded=false if the answer contains information
  that is not supported by the context.

- confidence must be between 0.0 and 1.0.
"""

    try:

        response = llm.invoke(prompt)

        raw = response_to_text(response)

        result = parse_json(raw)

        grounded = bool(
            result.get(
                "grounded",
                False
            )
        )

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        return {
            "grounded": grounded,
            "confidence_score": round(
                confidence * 100,
                2
            )
        }

    except Exception as error:

        print(
            "Grounding check error:",
            error
        )

        # Conservative fallback
        return {
            "grounded": False,
            "confidence_score": 0.0
        }


# =========================================================
# FINAL NODE
# =========================================================

def finalize_node(state: RAGState):

    if not state.get(
        "grounded",
        False
    ):

        return {
            "answer": REFUSAL
        }

    return {
        "answer": state.get(
            "answer",
            REFUSAL
        )
    }


# =========================================================
# LANGGRAPH
# =========================================================

workflow = StateGraph(
    RAGState
)

workflow.add_node(
    "retrieve",
    retrieve_node
)

workflow.add_node(
    "generate",
    generate_node
)

workflow.add_node(
    "grade",
    grade_node
)

workflow.add_node(
    "finalize",
    finalize_node
)


workflow.set_entry_point(
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "generate"
)

workflow.add_edge(
    "generate",
    "grade"
)

workflow.add_edge(
    "grade",
    "finalize"
)

workflow.add_edge(
    "finalize",
    END
)


rag_app = workflow.compile()


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def ask(question: str) -> dict:

    result = rag_app.invoke(
        {
            "question": question
        }
    )

    evidence = result.get(
        "evidence",
        []
    )

    chunks = [
        item["text"]
        for item in evidence
    ]

    return {
        "query": question,

        "final_answer": result.get(
            "answer",
            REFUSAL
        ),

        "retrieved_context_chunks": chunks,

        "confidence_score": result.get(
            "confidence_score",
            0.0
        ),
    }