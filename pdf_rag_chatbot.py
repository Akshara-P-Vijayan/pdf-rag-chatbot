from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
import os
from langchain_huggingface import HuggingFacePipeline

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)
def create_vector_store(chunks):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedder)
    return db
def get_hf_pipeline_llm():
    pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct",
                    tokenizer="tiiuae/falcon-7b-instruct",
                    max_new_tokens=200,
                    do_sample=True)
    return HuggingFacePipeline(pipeline=pipe)
def create_qa_chain(db):
    llm = get_hf_pipeline_llm()
    retriever = db.as_retriever(search_type="similarity", k=3)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa
file_path = "/content/drive/MyDrive/pdf-rag-chatbot/src/test1.pdf"
chunks = load_and_split_pdf(file_path)
db = create_vector_store(chunks)
qa_chain = create_qa_chain(db)

query = "What is this PDF about?"
response = qa_chain.invoke(query)

print(response)
