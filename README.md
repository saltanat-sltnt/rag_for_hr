# RAG for HR

RAG for HR is a chat assistant for answering employee questions based on provided HR/company documents. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant document chunks and generate answers based on them.

## Functionality

The project includes:

- loading PDF files from the `docs/` folder
- splitting documents into chunks using `CharacterTextSplitter`
- creating embeddings using OpenAI embeddings
- storing embeddings locally in a Chroma database
- retrieving relevant chunks from Chroma based on a user query
- generating answers using GPT-4o
- supporting chat history
- reformulating follow-up questions into standalone searchable queries
- running a continuous chat loop until the user types `quit`

## Project Structure

```text
rag_for_hr/
├── docs/
├── db/
├── ingestion_pipeline.py
├── retrieve_and_answer.py
├── history_aware_generation.py
├── requirements.txt
├── .gitignore
└── README.md