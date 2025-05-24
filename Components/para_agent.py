import os
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from together import Together
from langchain.llms.base import LLM
from typing import Any, List, Optional
from Components.para_utility import load_pdfs_from_file
from pydantic import PrivateAttr

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


class TogetherLLM(LLM):
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    temperature: float = 0.7  # Increased for more creative responses
    max_tokens: int = 1024  # Increased token length
    together_api_key: str = st.secrets["Together_API"]

    # Define private attribute
    client: Any = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Together(api_key=self.together_api_key)

    def _call(self, prompt: str, **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.9,  # Added for better response quality
            frequency_penalty=0.1,  # Added to reduce repetition
            presence_penalty=0.1  # Added to encourage diverse content
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "together_llm"


def split_docs(documents, chunk_size=500, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def initialize_model(documents=None):
    with st.spinner("Processing documents may take a few seconds"):
        # Initialize the LLM
        llm = TogetherLLM()
        
        if documents:
            new_pages = split_docs(documents)
            if not new_pages:
                st.error("No documents to process.")
                return None
                
            # Create vector store from documents
            db = Chroma.from_documents(
                new_pages,
                embedding_function,
                persist_directory="dataset"
            )
            db.persist()
            retriever = db.as_retriever(similarity_score_threshold=0.9)
        else:
            # For direct LLM queries without documents
            retriever = None

        prompt_template = """
        CONTEXT: {context}
        QUESTION: {question}"""

        PROMPT = PromptTemplate(template=f"[INST] {prompt_template} [/INST]", input_variables=["context", "question"])

        if retriever:
            # Use RetrievalQA chain for document-based queries
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=retriever,
                input_key='query',
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT},
                verbose=True
            )
        else:
            # Use simple LLM chain for direct queries
            from langchain.chains import LLMChain
            chain = LLMChain(
                llm=llm,
                prompt=PROMPT,
                verbose=True
            )
            
    st.success("Model initialized successfully!")
    return chain

class ConversationalAgent:
    def __init__(self, chain):
        self.chain = chain
        self.history = []

    def ask(self, query):
        context = " ".join([item['response'] for item in self.history])
        prompt_template = """
        CONTEXT: {context}
        QUESTION: {question}"""
        prompt = f"[INST] CONTEXT: {context} QUESTION: {query} {prompt_template} [/INST]"
        
        if hasattr(self.chain, 'retriever'):
            # For document-based queries
            response = self.chain(query)
            result = response['result']
            source_docs = response['source_documents']
        else:
            # For direct LLM queries
            result = self.chain.run(context=context, question=query)
            source_docs = []
            
        self.history.append({'query': query, 'response': result})
        return result, source_docs

def process_file(uploaded_file):
    """Process the uploaded file and return an agent"""

        
    documents = load_pdfs_from_file(uploaded_file)
    if documents is None:
        return None
    chain = initialize_model(documents)
    
    return ConversationalAgent(chain)

def demo_file_load():
    db=Chroma(
        persist_directory="dataset",
        embedding_function=embedding_function
    )
    
    llm = TogetherLLM()

    retriever = db.as_retriever(similarity_score_threshold=0.9)

    prompt_template = """
    CONTEXT: {context}
    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=f"[INST] {prompt_template} [/INST]", input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        input_key='query',
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=True
    )
    st.success("Response generated successfully!")
    return ConversationalAgent(chain)