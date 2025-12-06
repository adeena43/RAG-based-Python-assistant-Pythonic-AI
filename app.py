# app.py

import streamlit as st
from rag_engine import build_retriever, answer_question

st.set_page_config(
    page_title="Pythonic AI Assistant",
    page_icon="🐍",
    layout="centered"
)

# ---------- HEADER ----------
st.markdown(
    """
    <div style="text-align:center; padding:20px;">
        <h1 style="color:#4CAF50; font-size:42px;">🐍 Pythonic AI Assistant</h1>
        <p style="font-size:18px; color:#555;">Ask any Python / NumPy question.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- LOAD RAG RETRIEVER ----------
@st.cache_resource
def load_rag():
    return build_retriever()

retriever = load_rag()

# ---------- INPUT ----------
question = st.text_input("Ask your question:", "")

if st.button("Ask", use_container_width=True):
    if question.strip() == "":
        st.error("Please enter a valid question.")
    else:
        with st.spinner("🤖 Thinking..."):
            answer = answer_question(question, retriever)

        # ---------- ANSWER CARD ----------
        st.markdown(
            """
            <div style="
                background-color:#f8f9fa;
                padding:20px;
                border-radius:12px;
                box-shadow:0 0 10px rgba(0,0,0,0.1);
                margin-top:20px;
            ">
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"### ✨ Answer to: *{question}*")
        st.markdown(answer)

        st.markdown("</div>", unsafe_allow_html=True)
