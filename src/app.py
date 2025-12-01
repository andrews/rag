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


def query_documents(question, collection, n_results=4):
    # I'm assuming the OpenAI embedding function gets called here
    results = collection.query(
        # query_embeddings = [...]
        query_texts=question,
        n_results=n_results
    )

    # Extract relevant chunks
    # I'm assuming these are nested lists
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("=== Returning relevant chunks ===")
    return relevant_chunks

def generate_response(question, client, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following "
        "pieces of retrieved context to answer the question. If you don't know "
        "the answer, say that you don't know. Only use a few sentences and "
        "keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model = 'gpt-5-nano',
        messages = [
            {
                'role': 'system',
                'content': prompt,
            },
            {
                'role': 'user',
                'content': question
            }
        ]
    )
    # answer = response.choices[0].message
    return response

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
    # Comment below section out to avoid unnecessary embedding and upsert calls
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

    # Now we need to query the database
    # question = "Tell me which month I spent the most money and the kinds of things I spent it on."
    question = "I'm trying to save money. Tell me what kinds of purchases I should cut back on."

    relevant_chunks = query_documents(
        question=question,
        collection=collection
    )
    # With the chunks we retrieved, we can make the call to OpenAI
    # with our prompt, question, and relevant documents
    raw_answer = generate_response(
        question=question,
        client=client,
        relevant_chunks=relevant_chunks
    )

    print(raw_answer.choices[0].message.content)

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