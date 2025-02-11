

# As of 2/9/2025 We can only choose from (mistral,gemma2, qwen2) 
# Active developement is going on to support more LLM's via local api
# https://github.com/microsoft/graphrag/issues/657

# embedding model
# llama-server --port 8081 -m /mnt/models/model.file --temp 0.8 -ngl 99 --host 0.0.0.0 --embeddings --pooling mean --batch-size 8192 --ubatch-size 4096

# Regular model
# llama-server --port 8080 -m /mnt/models/model.file --temp 0.8 -ngl 99 --host 0.0.0.0

# make sure context size matches and that graphrag doesnt add more tokens than
# the embedding model can handle

import os
import streamlit as st
from ruamel.yaml import YAML
import subprocess
import fitz
import shutil
import requests

# Initialize YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

# Initialize session state
if 'uploaded_file_previous' not in st.session_state:
    st.session_state.uploaded_file_previous = None

if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

if 'user_query' not in st.session_state:
    st.session_state.user_query = ''  # Initialize the query input state
if 'last_submission' not in st.session_state:
    st.session_state.last_submission = ''  # To store the last successful query result

def init_folder():
    # Create folder
    subprocess.run(["mkdir", "-p", "./ragtest/input"], check=True)

def init_graphrag():
    # Initialize GraphRAG
    subprocess.run(["graphrag", "init", "--root", "./ragtest"], check=True)

def setup_yaml():
    # Initialize Services
    model_service = os.getenv("MODEL_ENDPOINT", "http://localhost:8001")
    embed_service = 'http://localhost:8002/v1'

    yaml_file = 'ragtest/settings.yaml'
    
    # Read the YAML file while preserving formatting
    with open(yaml_file, 'r') as f:
        settings_yaml = yaml.load(f)

    # Update the settings
    settings_yaml['llm']['model'] = "placeholder"
    settings_yaml['llm']['type'] = 'openai_chat'
    settings_yaml['llm']['max_tokens'] = 512
    settings_yaml['llm']['api_base'] = model_service
    settings_yaml['llm']['api_version'] = 'v1'
    settings_yaml['llm']['model_supports_json'] = False

    settings_yaml['embeddings']['llm']['type'] = 'openai_embedding'
    settings_yaml['embeddings']['llm']['max_tokens'] = 512
    settings_yaml['embeddings']['llm']['model'] = 'placeholder'
    settings_yaml['embeddings']['llm']['api_base'] = embed_service
    settings_yaml['embeddings']['llm']['api_version'] = 'v1'

    settings_yaml['chunks']['size'] = 1200

    # Perform health check on model and embed services
    try:
        request_model = requests.get(f'{model_service}/health', timeout=15)
        request_model.raise_for_status()  # Will raise an error for non-2xx responses
    except requests.RequestException as e:
        raise ValueError(f"Error connecting to model service: {e}")

    try:
        request_embed = requests.get(f'{embed_service}/health', timeout=15)
        request_embed.raise_for_status()  # Will raise an error for non-2xx responses
    except requests.RequestException as e:
        raise ValueError(f"Error connecting to embedding service: {e}")

    # Write the updated content back while preserving formatting
    with open(yaml_file, 'w') as f:
        yaml.dump(settings_yaml, f)

def pdf_to_text(pdf_path, output_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    text = ''
    
    # Extract text from each page
    for page in doc:
        text += page.get_text()
    
    # Save the extracted text to the output file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

def run():
    with st.spinner("Indexing the document with GraphRAG. This may take a moment..."):
        subprocess.run(
            ["graphrag", "index", "--root", "./ragtest"],
            stdout=subprocess.PIPE,  # Capture stdout
            stderr=subprocess.DEVNULL,  # Suppress stderr
            text=True
        )
    

def query(query):
    with st.spinner("Processing your query..."):
        query_result = subprocess.run(
            ["graphrag", "query", "--root", "./ragtest", "--method", "local", "--query", query],
            stdout=subprocess.PIPE,  # Capture stdout
            stderr=subprocess.DEVNULL,  # Suppress stderr
            text=True
        )
    output = query_result.stdout
    response_find = output.find("SUCCESS:")
    response = output[response_find + len("SUCCESS:"):].strip()
    st.session_state.last_submission = response  # Store the result in session state
    st.success("Query Complete!")
    st.write(response)

# Function to handle query submission and reset input field
def submit():
    st.session_state.user_query = st.session_state.widget
    st.session_state.widget = ''  # Reset the input field after submission

# Streamlit UI
st.title("GraphRAG PDF Query Interface")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.uploaded_file_previous:
        print("new pdf uploaded")
        st.session_state.uploaded_file_previous = uploaded_file.name
        # This part is only executed once when a new file is uploaded.
        print("startup")
        # If a new PDF is uploaded, delete the existing ragtest folder
        if os.path.exists("./ragtest"):
            shutil.rmtree("./ragtest", ignore_errors=True)
        # Save the uploaded file to a temporary location
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Initialize folders and GraphRAG
        init_folder()
        pdf_to_text("temp.pdf", "ragtest/input/document.txt")
        init_graphrag()
        setup_yaml()
        run()

        # Update session state
        st.session_state.rag_initialized = True

# If a PDF has been uploaded and GraphRAG is initialized, allow queries
if st.session_state.rag_initialized:
    print("Query")
    # Query input
    user_query = st.text_input("Enter your query:", key="widget", on_change=submit)
    
    # Automatically submit query after it is entered, but only if it's not empty
    if st.session_state.user_query.strip():  # Check if the query is not empty
        query(st.session_state.user_query)
