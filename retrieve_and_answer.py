from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
persistent_directory = "db/chroma_db"

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "Какая списочная численность персонала Группы КМГ была на конец 2024 года, и какое было распределение по полу?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"Query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")

# "What documents must an employee submit to HR on the joining date according to the Bennett University HR Manual?"
# "What are KMG’s main strategic goals and targets for 2031 related to sustainable development and reducing carbon intensity?",
# "Какая списочная численность персонала Группы КМГ была на конец 2024 года, и какое было распределение по полу?",
# "What are the four fundamental corporate values of KMG listed in the Code of Business Conduct?",
# "Какие основные цели кадровой политики КМГ указаны в разделе 4.1?"

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents".
"""

chat_model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assisstant."),
    HumanMessage(content=combined_input)
]

result = chat_model.invoke(messages)

print("\n --- Generated Response ---")
print(result.content)
