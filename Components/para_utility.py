import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import uuid  # For generating unique identifiers



def load_pdfs_from_file(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    st.write(documents[0].metadata["source"],"Documents in file path")
    return documents

def save_uploaded_file(uploaded_file, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # st.write(file_path,"path")
    return file_path


def load_pdfs_from_folder(folder_path):
    documents = []
    # print(os.listdir(folder_path))
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for i, doc in enumerate(docs):
                doc.metadata['title'] = filename
                doc.metadata['page'] = i + 1
            documents.extend(docs)
    return documents

# Example usage



# Add these functions after the existing imports
def create_user_storage():
    """Create user-specific storage directories if they don't exist"""
    # Create base directories
    user_dir = Path.home()/"chatbot_storage"
    pdf_dir = user_dir / "pdfs"
    embedding_dir = user_dir / "embeddings"
    
    # Create directories if they don't exist
    pdf_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    return pdf_dir, embedding_dir

def save_to_user_storage(uploaded_file):
    """Save uploaded file to user's local storage"""
    if uploaded_file is None:
        return None
        
    pdf_dir, _ = create_user_storage()
    file_path = pdf_dir / uploaded_file.name
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    return str(file_path)

# Modify the existing get_embedding_path function
def get_embedding_path(file_hash):
    """Get the path where embeddings should be stored"""
    _, embedding_dir = create_user_storage()
    return embedding_dir / f"{file_hash}_embeddings"

# Replace the existing file processing section with this updated version
