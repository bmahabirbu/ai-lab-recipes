import os
from ruamel.yaml import YAML
import subprocess
import fitz

def init_folder():
    # Create folder
    subprocess.run(["mkdir", "-p", "./ragtest/input"], check=True)

def init_graphrag():
    # Initialize GraphRAG
    subprocess.run(["graphrag", "init", "--root", "./ragtest"], check=True)

def setup_yaml():
    # Initialize Services
    model_service = os.getenv("MODEL_ENDPOINT", "http://localhost:8080")

    yaml_file = 'ragtest/settings.yaml'
    yaml = YAML()
    yaml.preserve_quotes = True  # Ensures quoted strings remain quoted
    yaml.indent(mapping=2, sequence=4, offset=2)  # Match original formatting

    # Read the YAML file while preserving formatting
    with open(yaml_file, 'r') as f:
        settings_yaml = yaml.load(f)

    # Update the settings
    settings_yaml['llm']['model'] = "placeholder"
    settings_yaml['llm']['type'] = 'openai_chat'
    settings_yaml['llm']['max_tokens'] = 512
    settings_yaml['llm']['api_base'] = model_service
    settings_yaml['llm']['api_version'] = 'v1'

    settings_yaml['embeddings']['llm']['type'] = 'openai_embedding'
    settings_yaml['embeddings']['llm']['max_tokens'] = 512
    settings_yaml['embeddings']['llm']['model'] = 'placeholder'
    settings_yaml['embeddings']['llm']['api_base'] = 'http://localhost:8081/v1'
    settings_yaml['embeddings']['llm']['api_version'] = 'v1'

    settings_yaml['chunks']['size'] = 1200

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

    print(f"Text extracted and saved to {output_path}")

def run():
    subprocess.run(["graphrag", "index", "--root", "./ragtest"], check=True)

def query(query):
    # Run the subprocess command to execute the query
    query_result = subprocess.run(
        ["graphrag", "query", "--root", "./ragtest", "--method", "local", "--query", query],
        capture_output=True, text=True
    )
    output = query_result.stdout
    print(output)


# init_folder()
# pdf_to_text("Resume.pdf", "ragtest/input/document.txt")
# init_graphrag()

# setup_yaml()
# run()
query("Do you think Brian is fit for a senior eng role or a entry level?")

# embedding model
# llama-server --port 8081 -m /mnt/models/model.file --temp 0.8 -ngl 99 --host 0.0.0.0 --embeddings --pooling mean --batch-size 8192 --ubatch-size 4096

# Regular model
# llama-server --port 8080 -m /mnt/models/model.file --temp 0.8 -ngl 99 --host 0.0.0.0

# make sure context size matches and that graphrag doesnt add more tokens than
# the embedding model can handle