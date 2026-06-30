from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "Какие основные цели кадровой политики КМГ указаны в разделе 4.1?"
# query = "Какие помидоры можно использовать для соления?"
print(f"Searching for: {query}\n")

# METHOD 1 - basic similarity search

# retriever = db.as_retriever(search_kwargs={"k": 3})
# docs = retriever.invoke(query)

# print(f"Retrieved {len(docs)} documents.")
# for i, doc in enumerate(docs, 1):
#     print(f"Doc {i}")
#     print(f"{doc.page_content}\n")

# print("-" * 60)

# METHOD 2 - similarity with score threshold

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.3
#     }
# )

# docs = retriever.invoke(query)
# print(f"Retrieved {len(docs)} documents (threshold = 0.3).")
# for i, doc in enumerate(docs, 1):
#     print(f"Doc {i}")
#     print(f"{doc.page_content}\n")

# print("-" * 60)

# METHOD 3 - MMR (Maximum Marginal Relevance)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,                 # final num of docs
        "fetch_k": 10,          # pool to choose from
        "lambda_mult": 0.5      # 0 = max diversity, 1 = max relevance
    }
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents (threshold = 0.3).")
for i, doc in enumerate(docs, 1):
    print(f"Doc {i}")
    print(f"{doc.page_content}\n")

print("-" * 60)
