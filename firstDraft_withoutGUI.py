import os
os.environ["OPENAI_API_KEY"] = "sk-proj-_J8kZHfh4xiGE0Q9Mnm-Kfu5Gqck2hOVehLbZLLCsCU_4CdqG-ImmT3BlbkFJRV-af9zAAH1zHHUuVXsv7Mk-MzK-bOvi7bYeGDkjT_jJcCst_w4gm39yOOdE7GmLbN49ECfh4A"
os.environ["USER_AGENT"] = "PythonicAI/1.0 (+adinafraz01@gmail.com)"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser

url = 'https://numpy.org/doc/2.3/user/absolute_beginners.html'
loader = WebBaseLoader(url)
docs = loader.load()
# print(f"Loaded {len(docs)} documents")
# print(docs[0].page_content)
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.split_documents(docs)

# chunk[0]

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

# print(vector_store.index_to_docstore_id)

retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":6})
# print(retriever)

# print(retriever.invoke("What is numpy?"))

llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template = """
                    You are a helpful python assistant. 
                    Answer ONLY from the provided text context.
                    Display answer in a beautiful and user understandable format.
                    If the context is insufficient, just say you don't know.

                    {context}
                    Question: {question}
                """,
                input_variables={'context', 'question'}
)


question = "how to initialize arrays in numpy, give one method briefly?"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

answer = llm.invoke(final_prompt)
print("Question: ",question)
print("Assistant's answer: ", answer.content)
