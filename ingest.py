import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions

def save_to_chroma(chunks):
    """
    Save document chunks to ChromaDB vector database.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Add documents to ChromaDB
    # This will automatically create the database if it doesn't exist
    # and persist it to the specified directory
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    return len(chunks)

def process_pdf(uploaded_file):
    # Save uploaded file temporarily
    temp_file = f"./temp_{uploaded_file.name}"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        loader = PyPDFLoader(temp_file)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(docs)
        return chunks
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    st.set_page_config(page_title="RAG Ingestion Tool", page_icon="📚")
    st.title("📚 PDF Ingestion for Voice Agent")
    st.write("Upload PDF documents to add them to the agent's knowledge base.")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button("Ingest Documents"):
            with st.spinner("Processing documents..."):
                total_chunks = 0
                for file in uploaded_files:
                    st.write(f"Processing {file.name}...")
                    chunks = process_pdf(file)
                    count = save_to_chroma(chunks)
                    total_chunks += count
                    st.success(f"✅ {file.name}: Added {count} chunks.")
                
                st.success(f"🎉 All done! Total {total_chunks} chunks added to ChromaDB.")

if __name__ == "__main__":
    main()
