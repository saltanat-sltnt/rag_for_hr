from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings)

chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

chat_history = []


def ask_question(user_question):
    if chat_history:
        messages = [SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")
                    ] + chat_history + [HumanMessage(content=f"New question: {user_question}")
                                        ]

        result = chat_model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 5})

    relevant_docs = retriever.invoke(search_question)

    print(f"Found {len(relevant_docs)} relevant documents:")
    for i, doc in enumerate(relevant_docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"Document {i}: {preview}...")

    combined_input = f"""Based on the following documents, please answer this question: {user_question}
    
    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in relevant_docs])}
    
    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "У меня недостаточно информации, чтобы ответить на этот вопрос.".
    """

    messages = [SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
                ] + chat_history + [HumanMessage(content=combined_input)
                                    ]

    result = chat_model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Response: {answer}")
    return answer


def start_chat():
    print("Starting chat... Type 'quit' to exit.")

    while True:
        question = input("\nEnter your question: ")

        if question.lower() == 'quit':
            print("Stopping the chat... Goodbye!")
            break

        ask_question(question)


def main():
    start_chat()


if __name__ == "__main__":
    main()
