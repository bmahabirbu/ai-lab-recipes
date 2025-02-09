import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
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

def main():

    graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

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

    embeddings= OpenAIEmbeddings(model="dummy", api_key="sk-no-key-required", base_url="http://localhost:8081/v1", use_function_response=True)

    text = """
    Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    She was, in 1906, the first woman to become a professor at the University of Paris.
    """

    documents = [Document(page_content=text)]

    # Transform documents into graph documents
    transformer = LLMGraphTransformer(llm=llm)
    graph_documents = transformer.convert_to_graph_documents(documents)

    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")

    graph.add_graph_documents(graph_documents)

    print(graph.schema)

    chain = GraphCypherQAChain.from_llm(
        llm, graph=graph, verbose=True
    )
    
    result = chain.invoke({"query": "who is Mariecurie"})

    print(result)

    
if __name__ == "__main__":
    main()