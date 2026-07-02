import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

from config import (
    PERSISTENT_DIRECTORY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCS_PATH
)

load_dotenv()


def load_documents(docs_path=DOCS_PATH):
    # load all pdf documents from docs directory
    print(f"--- Loading docuemnts from {docs_path}... ---")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No pdf files found in {docs_path}.")

    # show first 2 documents
    for i, doc in enumerate(documents[:2]):
        print(f"Document {i + 1}")
        print(f"Source: {doc.metadata['source']}")
        # print(f"    Content length: {len(doc.page_content)} characters")
        # print(f"    Content preview: {doc.page_content[:100]}...")
        # print(f"    Metadata: {doc.metadata}\n")

    return documents


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    # split documents into smaller chunks with overlap
    print("--- Splitting documents into chunks... ---")

    # text_splitter = CharacterTextSplitter(
    #     #separator=" ",  # Default separator. Other options include ["\n\n", "\n", ". ", " ", ""]
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    # )

    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],  # Multiple separators
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = recursive_splitter.split_documents(documents)

    if chunks:
        print(f"Split documents into {len(chunks)} chunks.")

    return chunks


def create_vector_store(chunks, persist_directory=PERSISTENT_DIRECTORY):
    # create and persist ChromaDB vector store

    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    print("--- Creating vector store... ---")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Main funtion.")

    # 1. load the files
    documents = load_documents()
    # 2. chunk the files
    chunks = split_documents(documents)
    # 3. emded and store in Vector DB
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()
