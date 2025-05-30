import streamlit as st
import os
import json
import zipfile
import tempfile
from Components.para_utility import load_pdfs_from_file, load_pdfs_from_folder, save_uploaded_file, save_to_user_storage,create_user_storage, get_embedding_path
from Components.para_agent import initialize_model, ConversationalAgent,process_file,demo_file_load
from Components.jd_generator import JDGenerator, JDInput, JDOutput
from Components.candidate_matcher import CandidateMatcher, Candidate
from transformers import pipeline
from Components.video_utility import save_uploaded_video, process_video_voice
import shutil
import atexit
import tempfile
import hashlib
from pathlib import Path
import nltk

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

# Page title
st.set_page_config(page_title='Resume Parser', page_icon='📄')
st.title('📄 Resume Parser & Job Matcher')

# Initialize session states
if 'job_description' not in st.session_state:
    st.session_state['job_description'] = None
if 'generated_jd' not in st.session_state:
    st.session_state['generated_jd'] = None
if 'jd_output' not in st.session_state:
    st.session_state['jd_output'] = None
if 'resumes' not in st.session_state:
    st.session_state['resumes'] = []
if 'parsed_resumes' not in st.session_state:
    st.session_state['parsed_resumes'] = []
if 'matched_candidates' not in st.session_state:
    st.session_state['matched_candidates'] = []
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 1

# Initialize JD Generator and Candidate Matcher
if 'jd_generator' not in st.session_state:
    st.session_state['jd_generator'] = JDGenerator()
if 'candidate_matcher' not in st.session_state:
    st.session_state['candidate_matcher'] = CandidateMatcher()

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows you to generate or upload a job description and process multiple resumes to extract key information and match candidates with the job requirements.')

    st.markdown('**How to use the app?**')
    st.warning('1. Generate or upload a job description\n2. Verify and edit the job description\n3. Upload one or more resume PDFs\n4. View the extracted information and job matches')

def process_zip_file(zip_file):
    """Process resumes from a zip file"""
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Get all PDF files from the extracted directory
        pdf_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        return pdf_files
    except Exception as e:
        st.error(f"Error processing zip file: {e}")
        return []

def extract_skills_from_jd(jd_text):
    """Extract skills from job description text"""
    chain = initialize_model()
    agent = ConversationalAgent(chain)
    response, _ = agent.ask("Extract all required skills from this job description. Return only the skills, one per line.")
    return [skill.strip() for skill in response.split('\n') if skill.strip()]

# Main content area
st.header('Step 1: Job Description Input')

# Job description input options
input_method = st.radio("Choose input method:", ["Text Input", "Generate JD"])

if input_method == "Text Input":
    job_description = st.text_area("Enter job description:", height=200)
    if job_description:
        st.session_state['job_description'] = job_description
        st.session_state['generated_jd'] = None
        st.session_state['jd_output'] = None
        st.session_state['current_step'] = 2

else:  # Generate JD
    st.subheader("Generate Job Description")
    
    # Basic Information
    role = st.text_input("Enter the job role/title:")
    experience = st.number_input("Years of experience required:", min_value=0, max_value=30, value=3)
    skills = st.text_area("Enter required skills (one per line):", height=100)
    
    # Additional Information
    with st.expander("Additional Information (Optional)"):
        location = st.text_input("Location:")
        company_type = st.selectbox("Company Type:", 
            ["", "Startup", "Enterprise", "Government", "Non-Profit", "Other"])
        industry = st.text_input("Industry:")
        salary_range = st.text_input("Salary Range:")
        additional_requirements = st.text_area("Additional Requirements:", height=100)
    
    if st.button("Generate JD", type="primary"):
        if role and skills:
            # Create JDInput object
            jd_input = JDInput(
                role=role,
                experience_years=experience,
                required_skills=[skill.strip() for skill in skills.split('\n') if skill.strip()],
                location=location if location else None,
                company_type=company_type if company_type else None,
                industry=industry if industry else None,
                salary_range=salary_range if salary_range else None,
                additional_requirements=additional_requirements if additional_requirements else None
            )
            
            try:
                # Generate JD
                jd_output = st.session_state['jd_generator'].generate_jd(jd_input)
                st.session_state['jd_output'] = jd_output
                
                # Format for display
                formatted_jd = st.session_state['jd_generator'].format_jd_for_display(jd_output)
                st.session_state['generated_jd'] = formatted_jd
                st.session_state['job_description'] = formatted_jd
                st.session_state['current_step'] = 2
                
                st.success("Job description generated successfully!")
            except Exception as e:
                st.error(f"Error generating JD: {e}")
        else:
            st.warning("Please provide both role and skills to generate JD")

# If we have a job description, show it and allow editing
if st.session_state['job_description']:
    st.header('Review Job Description')
    
    if st.session_state['generated_jd']:
        st.info("Review and edit the generated job description:")
        edited_jd = st.text_area("Edit job description:", value=st.session_state['generated_jd'], height=400)
        if edited_jd != st.session_state['generated_jd']:
            st.session_state['job_description'] = edited_jd
            st.session_state['generated_jd'] = edited_jd
    else:
        st.write(st.session_state['job_description'])

    # Resume Upload Section
    st.header('Step 2: Upload Resumes')
    st.info("You can upload multiple resumes either as individual PDFs or as a ZIP file containing PDFs.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_files = st.file_uploader("Upload individual PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file not in st.session_state['resumes']:
                    file_path = save_uploaded_file(uploaded_file, "dataset")
                    if file_path:
                        st.session_state['resumes'].append(uploaded_file)
                        st.success(f"Resume saved: {uploaded_file.name}")
    
    with col2:
        zip_file = st.file_uploader("Or upload a ZIP file containing PDFs", type=["zip"])
        if zip_file:
            pdf_files = process_zip_file(zip_file)
            for pdf_file in pdf_files:
                if pdf_file not in st.session_state['resumes']:
                    file_path = save_uploaded_file(pdf_file, "dataset")
                    if file_path:
                        st.session_state['resumes'].append(pdf_file)
                        st.success(f"Resume saved: {os.path.basename(pdf_file)}")

    # Process Resumes Button
    if st.session_state['resumes']:
        st.header('Step 3: Process Resumes')
        if st.button("Process Resumes and Find Matches", type="primary"):
            with st.spinner("Processing resumes and finding matches..."):
                resume_texts = []
                resume_paths = []
                for resume in st.session_state['resumes']:
                    file_path = save_uploaded_file(resume, "dataset")
                    if file_path:
                        documents = load_pdfs_from_file(file_path)
                        if documents:
                            resume_text = "\n".join(doc.page_content for doc in documents)
                            resume_texts.append(resume_text)
                            resume_paths.append(file_path)

                # Extract skills and required experience from job description
                jd_skills = []
                required_experience = 0
                
                if isinstance(st.session_state['job_description'], dict):
                    jd_skills = st.session_state['job_description'].get('required_skills', [])
                    required_experience = st.session_state['job_description'].get('experience_years', 0)
                else:
                    # Extract skills and experience from text
                    chain = initialize_model()
                    agent = ConversationalAgent(chain)
                    
                    # Extract skills
                    skills_response, _ = agent.ask("Extract all required skills from this job description. Return only the skills, one per line.")
                    jd_skills = [skill.strip() for skill in skills_response.split('\n') if skill.strip()]
                    
                    # Extract required experience
                    exp_response, _ = agent.ask("Extract the required years of experience from this job description. Return only the number.")
                    try:
                        required_experience = int(exp_response.strip())
                    except:
                        required_experience = 0

                # Match candidates
                matched_candidates = []
                for i, (resume_text, file_path) in enumerate(zip(resume_texts, resume_paths)):
                    try:
                        candidate = st.session_state['candidate_matcher']._extract_candidate_info(resume_text, file_path)
                        matched_candidates.append(candidate)
                    except Exception as e:
                        st.error(f"Error processing resume {i+1}: {str(e)}")
                        continue

                # Calculate match scores
                for candidate in matched_candidates:
                    # Create candidate embedding
                    candidate_embedding = st.session_state['candidate_matcher']._create_candidate_embedding(candidate)
                    
                    # Calculate overall similarity
                    similarity_score = st.session_state['candidate_matcher']._calculate_similarity(
                        st.session_state['candidate_matcher']._create_jd_embedding(st.session_state['job_description']),
                        candidate_embedding
                    )
                    print(similarity_score)
                    # Calculate skill match
                    skill_match, matching_skills = st.session_state['candidate_matcher']._calculate_skill_match_score(jd_skills, candidate.skills)
                    candidate.skill_match_percentage = skill_match
                    
                    # Calculate experience match
                    experience_match = st.session_state['candidate_matcher']._calculate_experience_match_score(required_experience, candidate.total_experience_years)
                    candidate.experience_match_score = experience_match
                    print(similarity_score,skill_match,experience_match)
                    # Combine scores (40% similarity, 40% skill match, 20% experience match)
                    candidate.match_score = (
                        similarity_score * 0.4 +
                        skill_match * 0.4 +
                        experience_match * 0.2
                    )   # Convert to percentage

                # Sort candidates by match score
                st.session_state['matched_candidates'] = sorted(matched_candidates, key=lambda x: x.match_score, reverse=True)

# Display matched candidates with enhanced information
if st.session_state['matched_candidates']:
    st.header('Candidate Matches')
    st.info("Candidates are ranked by their match score with the job description.")
    
    for i, candidate in enumerate(st.session_state['matched_candidates'], 1):
        with st.expander(f"Rank #{i} - {candidate.name} (Match Score: {candidate.match_score:.1f}%)"):
            st.markdown(st.session_state['candidate_matcher'].format_candidate_display(candidate))
            
            # Add a progress bar for each match component
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Skill Match", f"{candidate.skill_match_percentage:.1f}%")
            with col2:
                st.metric("Experience Match", f"{candidate.experience_match_score:.1f}%")
            with col3:
                st.metric("Overall Match", f"{candidate.match_score:.1f}%")

# Cleanup function
def cleanup_temp_files():
    """Clean up temporary files when the session ends"""
    try:
        if os.path.exists("dataset"):
            shutil.rmtree("dataset")
    except Exception as e:
        st.error(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup_temp_files)

uploaded_file=None
uploaded_video_file=None
agent=None

# with st.expander('About this app'):
#   st.markdown('**What can this app do?**')
#   st.info('This app allows users to upload a PDF file about a topic and get a Query response from Together LLM.')

#   st.markdown('**How to use the app?**')
#   st.warning('To engage with the app, go to the sidebar and upload a PDF or use the demo PDF. Send a Query and get your answer.')

# st.write("It may take a few minutes to generate query response.")

# # Initialize session state for demo mode
# if 'use_demo_pdf' not in st.session_state:
#     st.session_state['use_demo_pdf'] = False

# # Sidebar for accepting input parameters
# with st.sidebar:
#     st.header('1.1. Input data')
#     st.markdown('**1. Choose data source**')
    
#     # Add demo PDF option
#     use_demo = st.checkbox("Use demo PDF", value=st.session_state['use_demo_pdf'])
    
#     if not use_demo:
#         uploaded_file = st.file_uploader("Upload a pdf file", type=["pdf"])
#         if uploaded_file:
#             # Save to user's local storage
#             file_path = save_uploaded_file(uploaded_file, "dataset")
#             if file_path:
#                 st.success(f"File saved locally at: {file_path}")
#                 agent = process_file(file_path)
#             else:
#                 st.error("Failed to save file locally")
#     else:
#         st.success("Using demo PDF (size below 1MB is preferred)")

        
#         agent=demo_file_load()
        
#         st.session_state['use_demo_pdf'] = True
#     # Use session state to control the checkbox state
#     if 'generate_questions' not in st.session_state:
#         st.session_state['generate_questions'] = False

#     generate_questions_checkbox = st.checkbox("Generate 5 questions from the content", value=st.session_state['generate_questions'])
#     st.header('1.2. Upload Video')
#     uploaded_video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# if 'query_responses' not in st.session_state:
#     st.session_state['query_responses'] = []

# def add_query_response(query, response):
#     st.session_state.query_responses.append({'query': query, 'response': response})

# def summarize_text(text):
#     summarizer = pipeline("summarization")
#     summary = summarizer(text, max_length=200, min_length=30, do_sample=False)
#     return summary[0]['summary_text']

# col1, col2 = st.columns([4, 1])  # Create two columns with ratio 4:1
# with col1:
#     query = st.text_input("Enter your query", key="query_input")
# with col2:
#     submit_button = st.button("Enter", type="primary")  # Add a primary colored button

# # Reset the checkbox after the operation
# if submit_button and uploaded_file and generate_questions_checkbox:
#     query = f"Generate 5 flashcard questions based Context: {query}"
#     st.write(query)
#     if uploaded_file:
#         dataset_directory = "dataset"
#         file_path = save_uploaded_file(uploaded_file, dataset_directory)
#         documents = load_pdfs_from_file(file_path)
#         chain = initialize_model(documents)
#         agent = ConversationalAgent(chain)
#     if documents:
#         response, Source = agent.ask(query)
#         add_query_response(query, response)

#         # Uncheck the checkbox after processing
#         st.session_state['generate_questions'] = False

#         # Displaying the sources
#         for doc in Source:
#             page = doc.metadata['page']
#             snippet = doc.page_content[:200]
#             Source = {doc.metadata['source']}
#             source=Source.split('/')[-1]
#             Content = {doc.page_content[:50]}
#             st.write(doc.page_content)
        
#         if page:
#             st.write(response)
#             st.write("Data taken from source:", Source, " and page No: ", page)
#         if Content:
#             st.write("Taken content from:", Content)
#         query = ""
#     else:
#         st.write("No documents found.")

# # Modify the query processing section
# if submit_button and query and not uploaded_video_file:  # Check if button is pressed
#     if agent:
#         response, Source = agent.ask(query)
#         add_query_response(query, response)

#         # Displaying the sources
#         for doc in Source:
#             page = doc.metadata['page']
#             snippet = doc.page_content[:200]
#             Source = {doc.metadata['source']}
#             source=str(Source).split("/")[-1]
#             Content = {doc.page_content}
#             # print(Source)
        
#         if page:
#             st.write(response)
#             st.write("Data taken from source:", source, " and page No: ", page)
#         if Content:
#             st.write("Taken content from:", Content)
#         # Clear the query input after processing
#         # st.session_state.query_input = ""
#     else:
#         st.write("No documents found.")
# elif not query:
#     st.write("Enter query.")

# if  uploaded_video_file:
#     # Save the uploaded video file to a temporary location
#     try:
#         video_file_path = save_uploaded_video(uploaded_video_file)
        
#         # Process the video to extract voice and summarize
#         voice_text = process_video_voice(video_file_path)
#         # st.write("Voice Data:", voice_text)
#         st.markdown(f"**Voice Data:** <span style='font-size: 20px;'>{voice_text}</span>", unsafe_allow_html=True)
#         summary = summarize_text(voice_text)
#         # st.write("Voice Summary:", summary)
#         st.markdown(f"**Voice Summary:** <span style='font-size: 20px;'>{summary}</span>", unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"An error occurred: {e}")

# st.header('Previous Queries and Responses')

# if st.session_state.query_responses:
#     for i, qr in enumerate(st.session_state.query_responses, 1):
#         st.write(f"{i}. Query: {qr['query']}")
#         st.write(f"   Response: {qr['response']}")
# else:
#     st.write("No queries yet.")

# # Add this after session state initialization
# if 'needs_cleanup' not in st.session_state:
#     st.session_state.needs_cleanup = False

# def cleanup_temp_files():
#     """Clean up temporary files when the session ends"""
#     try:
#         # Clean up the temporary video files
#         temp_dir = tempfile.gettempdir()
#         for filename in os.listdir(temp_dir):
#             if filename.endswith(('.mp4', '.avi', '.mov', '.wav')):
#                 filepath = os.path.join(temp_dir, filename)
#                 os.remove(filepath)
                
#         # Clean up any temporary audio files in current directory
#         if os.path.exists("temp_audio.wav"):
#             os.remove("temp_audio.wav")
            
#         # Clean up dataset directory but keep user storage
#         if os.path.exists("dataset"):
#             shutil.rmtree("dataset")
            
#     except Exception as e:
#         st.error(f"Error during cleanup: {e}")

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('maxent_ne_chunker')
nltk.download('words')

