import os
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

# Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions

# Initialize Low-Temp Model for RAG (Temperature 0.2 as requested)
rag_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens=512,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

@tool
def database_search(query: str) -> str:
    """
    Search the internal knowledge base (database) for information from uploaded documents.
    Use this tool when the user asks about specific documents, manuals, or uploaded content.
    
    Args:
        query: The search query to find relevant information.
    
    Returns:
        Relevant context from the database.
    """
    try:
        logger.info(f"🔍 Searching database for: {query}")
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize Chroma client
        if not os.path.exists(CHROMA_PATH):
            return "❌ Database not found. Please upload documents using the ingestion app first."
            
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
        
        # Search for top 3 relevant chunks
        results = db.similarity_search(query, k=3)
        
        if not results:
            return f"❌ No relevant information found in the database for '{query}'."
            
        # Format output
        output = f"📚 Database Results for '{query}'\n\n"
        
        for idx, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content
            
            output += f"📄 *Source:* {os.path.basename(source)} (Page {page})\n"
            output += f"{content}\n\n"
            
        return output

    except Exception as e:
        logger.error(f"Error in database search: {e}")
        return f"❌ Database Search Error: {str(e)}"
