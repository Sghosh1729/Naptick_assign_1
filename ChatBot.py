from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from preprocessing import all_documents

class LangChainRetriever(BaseRetriever):
    base_retriever: Any = Field(...)  # Declare the field for Pydantic

    def _get_relevant_documents(self, query: str) -> List[Document]:
        if self.base_retriever.index is None or not self.base_retriever.texts:
            self.base_retriever.load_index()

        results = self.base_retriever.search(query, top_k=5)
        return [
            Document(page_content=r["text"], metadata=r["metadata"])
            for r in results
        ]


from VectorStore import VectorStoreRetriever



vec = VectorStoreRetriever()
vec.build_index(all_documents)
lcr = LangChainRetriever(base_retriever=vec)


# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "Enter_your_API_key"

# Initialize memory to track conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
# Load Gemini model via LangChain
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",  # Or try "gemini-pro" or other listed models
    temperature=0.7
)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=lcr,
    memory=memory,
    #verbose=True
)


def run_chat():
    print("Start chatting with your RAG system (type 'exit' to stop):")
    trivial_responses = ["thank you", "thanks", "ok", "got it", "cool"]

    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input in ["exit", "quit"]:
            break
        if user_input in trivial_responses:
            print("Bot: You're welcome!")
            continue
        response = rag_chain.run(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    run_chat()

#print(lcr.base_retriever)

