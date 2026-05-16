import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.title("AI Finance Assistant 💰")

api_key = st.text_input("Gemini API Key daalo:", type="password")

if api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=api_key)

    loader = PyMuPDFLoader("AI Finance Assistant - Problem Statement.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    Answer based on context only:
    {context}
    Question: {question}
    """)

    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    user_input = st.text_input("Sawaal poochho:")

    if user_input:
        response = chain.invoke(user_input)
        st.write("AI:", response.content[0]['text'])