import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
import cowsay


def load_documents_from_directory(directory_path):
    print("=== Loading documents from directory ===") # maybe some logging here?
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            fp = os.path.join(directory_path, filename)
            with open(fp, 'r', encoding='utf-8') as file:
                documents.append({'id': filename, 'text': file.read()})
    return documents

# Should we treat CSV files different?
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Chunking over newline characters
# Can an LLM effectively handle CSV data? Lots of formatting.
# There may exist better ways for LLMs to interact with CSV files and RAG
def split_csv(text, chunk_size=20, chunk_overlap=4):
    chunks = []
    start = 0
    lines = text.split('\n')
    while start < len(lines):
        end = start + chunk_size
        chunks.append(lines[start:end])
        start = end - chunk_overlap
    return chunks

if __name__ == '__main__':
    load_dotenv()

    openai_key = os.getenv('OPENAI_API_KEY')

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name='text-embedding-3-small'
    )

    # Initialize the Chroma client with persistence
    chroma_client = chromadb.PersistentClient(
        path='chroma_persistent_storage')
    collection_name = 'document_qa_collection'
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )

    client = OpenAI(api_key=openai_key)
    directory_path = './../bank-statement-data'

    # Hello world
    resp = client.chat.completions.create(
        model='gpt-5-nano',
        messages=[
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user',
             'content': "What is human life expentancy in the United states?"}
        ]
    )
    # Using a notebook would be nice... Wouldn't have to make the same API calls
    print(resp.choices[0].message.content)