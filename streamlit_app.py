import streamlit as st
import os
from Components.para_utility import load_pdfs_from_file, load_pdfs_from_folder, save_uploaded_file, save_to_user_storage,create_user_storage, get_embedding_path
from Components.para_agent import initialize_model, ConversationalAgent,process_file,demo_file_load
# from whisper import load_model  # Importing Whisper AI
from transformers import pipeline  # Ensure transformers is updated
from Components.video_utility import save_uploaded_video, process_video_voice
import shutil
import atexit
import tempfile
import hashlib
from pathlib import Path

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

# Page title
st.set_page_config(page_title='Ema Chatbot', page_icon='🤖')
st.title('🤖 Ema chatBot')

uploaded_file=None
uploaded_video_file=None
agent=None

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allows users to upload a PDF file about a topic and get a Query response from Together LLM.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and upload a PDF or use the demo PDF. Send a Query and get your answer.')

st.write("It may take a few minutes to generate query response.")

# Initialize session state for demo mode
if 'use_demo_pdf' not in st.session_state:
    st.session_state['use_demo_pdf'] = False

# Sidebar for accepting input parameters
with st.sidebar:
    st.header('1.1. Input data')
    st.markdown('**1. Choose data source**')
    
    # Add demo PDF option
    use_demo = st.checkbox("Use demo PDF", value=st.session_state['use_demo_pdf'])
    
    if not use_demo:
        uploaded_file = st.file_uploader("Upload a pdf file", type=["pdf"])
        if uploaded_file:
            # Save to user's local storage
            file_path = save_uploaded_file(uploaded_file, "dataset")
            if file_path:
                st.success(f"File saved locally at: {file_path}")
                agent = process_file(file_path)
            else:
                st.error("Failed to save file locally")
    else:
        st.success("Using demo PDF (size below 1MB is preferred)")

        
        agent=demo_file_load()
        
        st.session_state['use_demo_pdf'] = True
    # Use session state to control the checkbox state
    if 'generate_questions' not in st.session_state:
        st.session_state['generate_questions'] = False

    generate_questions_checkbox = st.checkbox("Generate 5 questions from the content", value=st.session_state['generate_questions'])
    st.header('1.2. Upload Video')
    uploaded_video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if 'query_responses' not in st.session_state:
    st.session_state['query_responses'] = []

def add_query_response(query, response):
    st.session_state.query_responses.append({'query': query, 'response': response})

def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=200, min_length=30, do_sample=False)
    return summary[0]['summary_text']

col1, col2 = st.columns([4, 1])  # Create two columns with ratio 4:1
with col1:
    query = st.text_input("Enter your query", key="query_input")
with col2:
    submit_button = st.button("Enter", type="primary")  # Add a primary colored button

# Reset the checkbox after the operation
if submit_button and uploaded_file and generate_questions_checkbox:
    query = f"Generate 5 flashcard questions based Context: {query}"
    st.write(query)
    if uploaded_file:
        dataset_directory = "dataset"
        file_path = save_uploaded_file(uploaded_file, dataset_directory)
        documents = load_pdfs_from_file(file_path)
        chain = initialize_model(documents)
        agent = ConversationalAgent(chain)
    if documents:
        response, Source = agent.ask(query)
        add_query_response(query, response)

        # Uncheck the checkbox after processing
        st.session_state['generate_questions'] = False

        # Displaying the sources
        for doc in Source:
            page = doc.metadata['page']
            snippet = doc.page_content[:200]
            Source = {doc.metadata['source']}
            source=Source.split('/')[-1]
            Content = {doc.page_content[:50]}
            st.write(doc.page_content)
        
        if page:
            st.write(response)
            st.write("Data taken from source:", Source, " and page No: ", page)
        if Content:
            st.write("Taken content from:", Content)
        query = ""
    else:
        st.write("No documents found.")

# Modify the query processing section
if submit_button and query and not uploaded_video_file:  # Check if button is pressed
    if agent:
        response, Source = agent.ask(query)
        add_query_response(query, response)

        # Displaying the sources
        for doc in Source:
            page = doc.metadata['page']
            snippet = doc.page_content[:200]
            Source = {doc.metadata['source']}
            source=str(Source).split("/")[-1]
            Content = {doc.page_content}
            # print(Source)
        
        if page:
            st.write(response)
            st.write("Data taken from source:", source, " and page No: ", page)
        if Content:
            st.write("Taken content from:", Content)
        # Clear the query input after processing
        # st.session_state.query_input = ""
    else:
        st.write("No documents found.")
elif not query:
    st.write("Enter query.")

if  uploaded_video_file:
    # Save the uploaded video file to a temporary location
    try:
        video_file_path = save_uploaded_video(uploaded_video_file)
        
        # Process the video to extract voice and summarize
        voice_text = process_video_voice(video_file_path)
        # st.write("Voice Data:", voice_text)
        st.markdown(f"**Voice Data:** <span style='font-size: 20px;'>{voice_text}</span>", unsafe_allow_html=True)
        summary = summarize_text(voice_text)
        # st.write("Voice Summary:", summary)
        st.markdown(f"**Voice Summary:** <span style='font-size: 20px;'>{summary}</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.header('Previous Queries and Responses')

if st.session_state.query_responses:
    for i, qr in enumerate(st.session_state.query_responses, 1):
        st.write(f"{i}. Query: {qr['query']}")
        st.write(f"   Response: {qr['response']}")
else:
    st.write("No queries yet.")

# Add this after session state initialization
if 'needs_cleanup' not in st.session_state:
    st.session_state.needs_cleanup = False

def cleanup_temp_files():
    """Clean up temporary files when the session ends"""
    try:
        # Clean up the temporary video files
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.endswith(('.mp4', '.avi', '.mov', '.wav')):
                filepath = os.path.join(temp_dir, filename)
                os.remove(filepath)
                
        # Clean up any temporary audio files in current directory
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
            
        # Clean up dataset directory but keep user storage
        if os.path.exists("dataset"):
            shutil.rmtree("dataset")
            
    except Exception as e:
        st.error(f"Error during cleanup: {e}")

