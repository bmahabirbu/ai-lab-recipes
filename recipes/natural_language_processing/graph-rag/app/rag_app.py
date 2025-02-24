import sys
import os
import shutil
import nest_asyncio
import streamlit as st
import fitz

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

WORKING_DIR = "rag_data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "dummy"
API_KEY = "dummy"

# model_service = os.getenv("MODEL_ENDPOINT",
#                           "http://localhost:8001")

model_service = os.getenv("MODEL_ENDPOINT",
                          "http://localhost:8080/v1")

# Check if folder exists
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Initiate rag class
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=model_service,
        api_key=API_KEY,
        **kwargs,
    )

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    llm_model_max_token_size=2000,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)

# Initialize session state
if 'uploaded_file_previous' not in st.session_state:
    st.session_state.uploaded_file_previous = None

if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

if 'user_query' not in st.session_state:
    st.session_state.user_query = ''
if 'last_submission' not in st.session_state:
    st.session_state.last_submission = ''

def pdf_to_text(pdf_path, output_path):
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page in doc:
            text += page.get_text()
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        raise

def query(query, mode="mix"):
    print('\n')
    print("query: ",query)
    try:
        with st.spinner("Processing your query..."):
            response = rag.query(query, param=QueryParam(mode=mode))
        
        print('\n')
        print("response: ",response)
        st.session_state.last_submission = response
        st.success("Query Complete!")
        st.write(response)
    except Exception as e:
        st.error(f"Error processing query: {e}")

# Streamlit UI
st.title("GraphRAG PDF Query Interface")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.uploaded_file_previous:
        st.session_state.uploaded_file_previous = uploaded_file.name
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR, ignore_errors=True)
        os.makedirs(WORKING_DIR)

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            with st.spinner("Processing PDF..."):
                pdf_to_text("temp.pdf", "document.txt")
                with open("document.txt", "r", encoding="utf-8") as f:
                    rag.insert(f.read())
            st.session_state.rag_initialized = True
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

if st.session_state.rag_initialized:
    query_mode = st.radio(
        "Select query mode:",
        options=["local", "global", "naive", "mix"],
        index=3,
        key="mode"
    )
    st.session_state.query_mode = query_mode

    # Use a unique key for the text input to avoid conflicts
    user_query = st.text_input("Enter your query:", key="query_input")

    if st.button("Submit"):
        if user_query.strip():
            st.session_state.user_query = user_query
            query(st.session_state.user_query, mode=st.session_state.query_mode)