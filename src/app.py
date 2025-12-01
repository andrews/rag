import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
import cowsay


def load_documents_from_directory(directory_path, file_extension='.txt'):
    print("=== Loading documents from directory ===") # maybe some logging here?
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(file_extension):
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
        # Rest of program is expecting a string...
        # Joining lines into one string for now
        chunks.append('\n'.join(lines[start:end]))
        start = end - chunk_overlap
    return chunks


def create_csv_chunks(docs):
    chunked_csv_files = []
    for doc in docs:
        chunks = split_csv(doc['text'])
        print("=== Splitting bank statements into chunks ===")
        for i, chunk in enumerate(chunks):
            chunked_csv_files.append({'id': f'{doc['id']}_chunk{i+1}', 'text':chunk})
    return chunked_csv_files


def get_openai_embedding(client, text):
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    embedding = response.data[0].embedding
    print("=== Generating embeddings... ===")
    return embedding


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
    
    directory_path = './bank-statement-data'
    docs = load_documents_from_directory(directory_path, file_extension='.csv')
    chunked_csv_files = create_csv_chunks(docs)
    
    # Generate embeddings for bank statement chunks
    for chunk in chunked_csv_files:
        chunk["embedding"] = get_openai_embedding(client, chunk["text"])

    # Upsert embeddings into ChromaDB
    for chunk in chunked_csv_files:
        print("=== Inserting chunks into db ===")
        collection.upsert(
            ids=[chunk['id']],
            documents=[chunk['text']],
            embeddings=[chunk['embedding']]
        )

    # So we can stay in the program and poke around
    while True:
        cmd = input()
        if cmd == "done":
            break
        try:
            exec(cmd)
        except:
            print("Invalid command. Try again")
            continue

    # # Hello world
    # resp = client.chat.completions.create(
    #     model='gpt-5-nano',
    #     messages=[
    #         {'role': 'system', 'content': "You are a helpful assistant."},
    #         {'role': 'user',
    #          'content': "What is human life expentancy in the United states?"}
    #     ]
    # )
    # # Using a notebook would be nice... Wouldn't have to make the same API calls
    # print(resp.choices[0].message.content)