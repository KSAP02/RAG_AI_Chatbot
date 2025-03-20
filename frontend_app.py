import streamlit as st
import importlib
from dotenv import load_dotenv
from backend import ChatbotBackend
import os

# Set page configuration
st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Get the Hugging Face API from .env file
load_dotenv()
api_key = os.getenv('HF_API_KEY')
# can also create secrets.toml file and use st.secrets to retrieve api keys

# Initialize Session state variables

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = ChatbotBackend(api_key)
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] 
       
# Main layout
st.title("Document Chabot")

# Sidebar with document upload
with st.sidebar:
    st.header("Upload Documents")
    
    uploaded_file = st.file_uploader("Add documents to chat with",
                                     type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing document..."):
            # Extract text from document
            
            # Avoiding top level code(adding imports at the top),
            # so the same lib/file is not imported everytime streamlit is interacted with
            
            # Dynamically retrieveing the function without explicitly specifying at the top imports
            document_processor = importlib.import_module("document_processor")
            documents = document_processor.extract_text_from_file(uploaded_file)
            
            if documents:
                document_text = documents[0].page_content
            
            # Chunk the document text
            chunks, metadatas = document_processor.chunk_text(documents)
            st.write("Received the chunks")
            
            # Add documents to the chatbot
            st.session_state.chatbot.create_vector_store(chunks, metadatas)
            # st.write(len(embeddings))
            st.success(f"Document processed: {uploaded_file.name}")
            
    # Settings
    st.header("Settings")
    use_context = st.checkbox("Use Document Context", value=True)
    
    # About Section
    st.header("About")
    st.write('''
             This Chatbot uses:
            - T5/FLAN-Base for responses
            - Sentence Transformers for embeddings
            - FAISS for vector search
            ''')

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask me something...")

if user_input:
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
        
    # Get response from the chatbot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.get_answer(
                user_input,
                chunks,
                use_context=use_context
            )
        
        st.write(response)
        
    # Add Assistent response to History
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })