import google.generativeai as genai
from VectorStore import VectorStoreRetriever  # your retriever class
import os
from preprocessing import all_documents

class GeminiRAGPipeline:
    def __init__(self, retriever: VectorStoreRetriever, model_name="gemini-1.5-pro-001", api_key=None):
        self.retriever = retriever

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Gemini API key not set. Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=self.api_key, transport="rest")
        self.model = genai.GenerativeModel(model_name)

    def generate_answer(self, query, collection_filter=None, top_k=3):
        # Step 1: Retrieve documents
        retrieved_docs = self.retriever.search(query, collection_filter=collection_filter, top_k=top_k)

        if not retrieved_docs:
            return "Sorry, I couldn't find relevant information."

        context = "\n".join([doc["text"] for doc in retrieved_docs])

        # Step 2: Format prompt
        prompt = f"""You are a helpful assistant with access to user data.
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

        # Step 3: Send to Gemini
        response = self.model.generate_content(prompt)
        return response.text
#print("CHECK")

retriever = VectorStoreRetriever()
retriever.build_index(all_documents)  # Your loaded docs

rag = GeminiRAGPipeline(retriever)
query = "How much did I sleep last night?"

response = rag.generate_answer(query, collection_filter="wearable")
print("BOT:", response)