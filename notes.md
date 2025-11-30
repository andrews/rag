# Retrieval-Augmented Generation (RAG)
Learning more about RAGs and creating a sample project from scratch.

Currently following a video from freeCodeCamp __RAG Fundamentals and Advanced Techniques__, link [here](https://www.youtube.com/watch?v=ea2W8IogX80).

## Notes
  - Components:
    - Retriever: Identifies and retrieves relevant documents
    - Generator: Takes retrieved docs and the input query to generate coherent and contextually relevant responses.
  - Combining retrieval-based systems and generation-based models to produce conextually relevant responses
    - _Efficient way to customize an LLM with your own data_. Injecting your own data into LLM, in addition to what it was trained on
  - Overview
    1. Have your documents
        - Parsing + preprocessing
    2. Gets chunked
        - Chunking
    3. Goes through an _embedding LLM_, to create embeddings
        - Created vectors from the chunks (vectorizing)
        - Saved into a _vector store_, database (__indexing__).
    4. The query also goes through the embedding LLM. The query embeddings get compared to the embeddings from the documents you provided.
        - Vectorized, then sent into the vector store for search
        - Retrieved
    5. The most similar embeddings to your query embeddings get pushed together through a _general LLM_, which then generates a response. The _generated_ response is _augmented_ by the data that was _retrieved_.
        - Relevant docs together with a prompt and the query get passed througj LLM, then response is generated and returned.

  - Special packages: openai, chromadb, python-dotenv