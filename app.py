import json,streamlit as st
from rag_pipeline import ask
st.set_page_config(page_title="Agentic AI Gemini RAG",page_icon="🤖",layout="wide")
st.title("🤖 Agentic AI — Gemini RAG Chatbot")
st.caption("Gemini Embeddings + Pinecone + LangGraph + Gemini LLM")
q=st.text_input("Ask a question about the Agentic AI eBook:")
if st.button("Submit",type="primary") and q.strip():
    with st.spinner("Retrieving and generating grounded answer..."):
        try:
            result=ask(q.strip())
            st.subheader("Answer"); st.write(result["final_answer"])
            c=float(result["confidence_score"]); st.subheader("Confidence"); st.progress(min(max(c/100,0),1)); st.write(f"{c:.2f}%")
            with st.expander("Retrieved Context Chunks"):
                for i,x in enumerate(result["retrieved_context_chunks"],1): st.markdown(f"**Chunk {i}**\n\n{x}\n\n---")
            with st.expander("Required JSON Response"): st.code(json.dumps(result,indent=2),language="json")
        except Exception as e: st.error(str(e)); st.exception(e)
