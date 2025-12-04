"""
Streamlit UI for Voice Agent
Combined chat interface and document ingestion
"""

import os
import base64
import streamlit as st
from dotenv import load_dotenv
from scripts.agent import agent, agent_config
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Page configuration
st.set_page_config(
    page_title="Samantha",
    page_icon="🌀",
    layout="centered"
)

# Title and description
# Encode image to base64 for inline display
with open("assets/fastrtc.png", "rb") as img_file:
    img_data = base64.b64encode(img_file.read()).decode()

st.markdown(
    f"""
    <h1 style='text-align:center'>
        <img src="data:image/png;base64,{img_data}" width="70" style="vertical-align:middle; margin-right: 10px;">
        <span style='color:#fa2dab;'>Ur's AI Assistant Samantha</span>
    </h1>
    """,
    unsafe_allow_html=True
)

# Create tabs
tab1, tab2 = st.tabs(["💬 Chat", "📚 Upload Documents"])

# ==================== TAB 1: CHAT ====================
with tab1:
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Chat input at the top - in the same line with Send button
    col_input, col_button = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_input(
            "Ask me anything...", 
            key="chat_input",
            placeholder="Type your message here...",
            label_visibility="collapsed"
        )
    
    with col_button:
        send_button = st.button("Send 📤", use_container_width=True, type="primary")
    
    # Process input when send button is clicked
    if send_button and user_input.strip() and not st.session_state.processing:
        st.session_state.processing = True
        prompt = user_input.strip()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get agent response
        try:
            # Invoke the agent
            agent_reply = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=agent_config,
            )
            
            # Extract the response
            response = agent_reply["messages"][-1].content
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.session_state.processing = False
        # Clear input and rerun
        st.rerun()
    
    st.divider()
    
    # Display chat history in a container with fixed height
    chat_container = st.container(height=500)
    with chat_container:
        if len(st.session_state.messages) == 0:
            st.info("👋 Start a conversation with Samantha!")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

# ==================== TAB 2: DOCUMENT INGESTION ====================
with tab2:
    st.markdown("### 📄 Upload PDF Documents")
    st.write("Add documents to Samantha's knowledge base for enhanced responses.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload one or more PDF documents to add to the knowledge base"
    )

    if uploaded_files:
        if st.button("🚀 Ingest Documents", type="primary"):
            with st.spinner("Processing documents..."):
                total_chunks = 0
                
                for file in uploaded_files:
                    st.write(f"📖 Processing **{file.name}**...")
                    
                    # Save uploaded file temporarily
                    temp_file = f"./temp_{file.name}"
                    with open(temp_file, "wb") as f:
                        f.write(file.getbuffer())
                    
                    try:
                        # Load PDF
                        loader = PyPDFLoader(temp_file)
                        docs = loader.load()
                        
                        # Split into chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            add_start_index=True,
                        )
                        chunks = text_splitter.split_documents(docs)
                        
                        # Save to ChromaDB
                        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                        Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=CHROMA_PATH
                        )
                        
                        count = len(chunks)
                        total_chunks += count
                        st.success(f"✅ **{file.name}**: Added {count} chunks")
                        
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                
                st.success(f"🎉 **All done!** Total {total_chunks} chunks added to ChromaDB.")
                st.balloons()


# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("ℹ️ About Samantha")
    st.markdown("""
    Your AI assistant with access to:
    
    - 🔍 **Web Search** (Tavily)
    - 📈 **Stock Information** (YFinance)
    - 🌤️ **Weather Data**
    - ✈️ **Flight Search**
    - 🏨 **Hotel Search**
    - 📚 **Document Database** (ChromaDB)
    
    Powered by **Cerebras LLM** for fast responses.
    """)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    
    # Display agent configuration
    st.divider()
    st.subheader("⚙️ Configuration")
    st.text(f"Thread ID: {agent_config['configurable']['thread_id']}")
    st.text(f"Chat Messages: {len(st.session_state.messages) if 'messages' in st.session_state else 0}")