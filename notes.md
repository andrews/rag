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

## Building the app
  - Used `.gitignore` for the environment variables (OpenAI key) and the bank statement data.
    - Also used `.gitkeep` in that folder, so we still _see_ it, it's committed to the repo, but the contents aren't.
  - Using virtual environments and added a `requirements.txt` file. Also added the `.venv` folder to `.gitignore` since we don't need everything committed to git.
  - Loaded the API key with `python-dotenv`
  - Made a call using OpenAI API
  - Started chunking bank statement files.
    - The tutorial is chunking news articles, with overlap on the characters. However, this may not make much sense with CSV files. Might be more useful to chunk CSV files by lines instead of by characters. Will try both ways to see how it affects results.
    - Ran into some issues. The original chunking function creates string chunks. The CSV function I wrote created list chunks, where each element of the list was a separate line item in the CSV. I joined them (with `'\n'`) for now. 
  - Created embeddings using OpenAI API and upserted them into the Chroma DB.

## Thoughts
  - Are there better ways for LLMs to interact with CSV or tabular data? CSV files contain lots of formatting. Is it token efficient? LLM fine tuning on CSV files? Different way to ingest CSV files?
  - We are directly calling the embedding function on the text. When we created the Chroma DB, we specified an embedding function and the API key. How do these two things interact? Is the embedding happening twice? 
    - [Link](https://docs.trychroma.com/docs/collections/manage-collections#embedding-functions) to Chroma documentation, when specifying the embedding function in your collection. Not conclusive.
    - According to [here](https://docs.trychroma.com/docs/collections/update-data), the collection's embedding function will only be used if no `embeddings` are provided. We provide the embeddings, so no duplicate computations are happening.