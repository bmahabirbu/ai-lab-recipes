import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
import streamlit as st
import tempfile
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

def initialize_neo4j_connection(url, username, password):
    """Initialize and return a Neo4j graph connection."""
    try:
        graph = Neo4jGraph(url=url, username=username, password=password)
        st.sidebar.success("Connected to Neo4j database.")
        return graph
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None

def process_pdf(uploaded_file, graph, llm, embeddings, url, username, password):
    """Process the uploaded PDF, extract entities and relationships, and upload to Neo4j."""
    with st.spinner("Processing the PDF..."):
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load and split the PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        docs = text_splitter.split_documents(pages)

        # Prepare documents for graph transformation
        lc_docs = [
            Document(page_content=doc.page_content.replace("\n", ""), metadata={'source': uploaded_file.name})
            for doc in docs
        ]

        # Clear the graph database
        graph.query("MATCH (n) DETACH DELETE n;")

        # Transform documents into graph documents
        transformer = LLMGraphTransformer(llm=llm)
        graph_documents = transformer.convert_to_graph_documents(lc_docs)

        # Add the generated graph into Neo4j
        graph.add_graph_documents(graph_documents)

        print(graph_documents)

        # Create a vector index for the graph
        Neo4jVector.from_existing_graph(
            embeddings,
            url=url,
            username=username,
            password=password,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )

        st.success(f"{uploaded_file.name} processing is complete.")

def main():
    st.set_page_config(
        layout="wide",
        page_title="Graphy v1",
        page_icon=":graph:"
    )

    # Initialize LLM and embeddings
    model_service = os.getenv("MODEL_ENDPOINT", "http://localhost:8080")
    model_service = f"{model_service}/v1"
    model_service_bearer = os.getenv("MODEL_ENDPOINT_BEARER")
    model_name = "gpt-4o"  # Adjust the model name as needed

    llm = ChatOpenAI(
        base_url=model_service,
        api_key="sk-no-key-required" if model_service_bearer is None else model_service_bearer,
        model=model_name,
    )

    if 'embeddings' not in st.session_state:
        st.session_state['embeddings'] = OpenAIEmbeddings(model="-", api_key="sk-no-key-required", base_url="http://localhost:8081/v1")

    if 'llm' not in st.session_state:
        st.session_state['llm'] = llm

    embeddings = st.session_state['embeddings']
    llm = st.session_state['llm']

    # Neo4j connection details
    neo4j_url = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "yourpassword"

    # Initialize Neo4j connection
    if 'neo4j_connected' not in st.session_state:
        graph = initialize_neo4j_connection(neo4j_url, neo4j_username, neo4j_password)
        if graph:
            st.session_state['graph'] = graph
            st.session_state['neo4j_connected'] = True
    else:
        graph = st.session_state.get('graph', None)

    # Ensure that the Neo4j connection is established before proceeding
    if graph:
        uploaded_file = st.file_uploader("Please select a PDF file.", type="pdf")

        if uploaded_file is not None and 'qa' not in st.session_state:
            process_pdf(uploaded_file, graph, llm, embeddings, neo4j_url, neo4j_username, neo4j_password)

            # Retrieve the graph schema
            graph.refresh_schema()
            st.write("Graph Schema:", graph.schema)

            # Define the prompt template for Cypher query generation
            template = """
            Task: Generate a Cypher statement to query the graph database.

            Instructions:
            - Use only the relationship types and properties provided in the schema.
            - Do not use any relationship types or properties that are not provided in the schema.
            - Ensure the generated Cypher statement is syntactically correct and valid.

            Schema:
            {schema}

            Note: 
            - Do not include explanations or apologies in your answers.
            - Do not answer questions that ask anything other than creating Cypher statements.
            - Do not include any text other than the generated Cypher statement.

            Question: {question}""" 

            question_prompt = PromptTemplate(
                template=template, 
                input_variables=["schema", "question"] 
            )

            # Initialize the QA chain
            qa = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                cypher_prompt=question_prompt,
                verbose=True,
                allow_dangerous_requests=True
            )
            st.session_state['qa'] = qa
    else:
        st.warning("Failed to connect to the Neo4j database. Please check the connection details.")

    # Allow users to ask questions
    if 'qa' in st.session_state:
        st.subheader("Ask a Question")
        with st.form(key='question_form'):
            question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and question:
            with st.spinner("Generating answer..."):
                try:
                    res = st.session_state['qa'].invoke({"query": question})
                    st.write("\n**Answer:**\n" + res['result'])
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()