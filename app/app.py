import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile

st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

@st.cache_resource
def load_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding)
    
    gen_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", tokenizer="tiiuae/falcon-7b-instruct", max_new_tokens=200)
    llm = HuggingFacePipeline(pipeline=gen_pipeline)
    
    retriever = db.as_retriever(search_type="similarity", k=3)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        chain = load_chain(tmp.name)

    query = st.text_input("Ask a question about the PDF:")
    if query:
        answer = chain.invoke(query)
        st.write("🤖", answer)
