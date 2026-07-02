from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from config import PERSISTENT_DIRECTORY, EMBEDDING_MODEL
from dotenv import load_dotenv

load_dotenv()


def load_vectorstore():
    # load existing Chroma db
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    db = Chroma(
        persist_directory=PERSISTENT_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return db


def get_similarity_retriever(db, k=5):
    # basic vector similarity search
    return db.as_retriever(
        search_kwargs={"k": k}
    )


def get_threshold_retriever(db, k=5, score_threshold=0.3):
    # vector search with score threshold
    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold
        }
    )


def get_mmr_retriever(db, k=5, fetch_k=20, lambda_mult=0.5):
    # vector seach, balances relevance and diversity
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,                     # final num of docs
            "fetch_k": fetch_k,         # pool to choose from
            "lambda_mult": lambda_mult  # 0 = max diversity, 1 = max relevance
        }
    )


def get_all_documents_from_chroma(db):
    # extract all stored documents from Chroma for BM25 keyword search
    data = db.get()

    documents = []
    for text, metadata in zip(data["documents"], data["metadatas"]):
        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    return documents


def get_bm25_retriever(db, k=5):
    # keyword-based BM25 retriever
    documents = get_all_documents_from_chroma(db)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    return bm25_retriever


def get_hybrid_retriever(db, k=5, vector_weight=0.5, keyword_weight=0.5):
    # hybrid search: vector search + keyword BM25 search

    vector_retriever = get_mmr_retriever(
        db,
        k=k,
        fetch_k=10,
        lambda_mult=0.5
    )

    bm25_retriever = get_bm25_retriever(db, k=k)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, keyword_weight]
    )

    return hybrid_retriever


# def test_retriever():
#     db = load_vectorstore()

#     # query = "Какие основные цели кадровой политики КМГ указаны в разделе 4.1?"
#     query = "Сколько времени занимает выполнение мероприятия «Карта возможностей»?"
#     print(f"Searching for: {query}\n")

#     # Choose one retriever here
#     retriever = get_hybrid_retriever(db, k=5)
#     # retriever = get_similarity_retriever(db, k=5)
#     # retriever = get_mmr_retriever(db, k=5)
#     # retriever = get_threshold_retriever(db, k=5, score_threshold=0.3)

#     docs = retriever.invoke(query)

#     print(f"Retrieved {len(docs)} documents.")
#     for i, doc in enumerate(docs, 1):
#         print(f"\nDoc {i}")
#         print(doc.page_content[:1000])


# if __name__ == "__main__":
#     test_retriever()
