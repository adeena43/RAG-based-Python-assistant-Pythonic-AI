# rag_engine.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate

os.environ["USER_AGENT"] = "PythonicAI/1.0 (+adinafraz01@gmail.com)"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --------- LOAD AND BUILD VECTOR STORE (one time only) ---------
def build_retriever():
    url = "https://numpy.org/doc/2.3/user/absolute_beginners.html"

    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":6})
    return retriever


# --------- LLM + PROMPT ---------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
You are a helpful Python documentation assistant.
Answer ONLY using the context below.
Format the answer in clean markdown with headings and bullet points.
If context does not contain the answer, say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}
""",
    input_variables=["context", "question"]
)


# --------- MAIN FUNCTION CALLED FROM STREAMLIT ---------
def answer_question(question, retriever):
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({"context": context_text, "question": question})
    answer = llm.invoke(final_prompt)

    return answer.content
