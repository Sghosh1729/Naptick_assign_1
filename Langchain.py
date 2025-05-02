from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from VectorStore import VectorStoreRetriever


class LangChainRetriever(BaseRetriever):
    def __init__(self, base_retriever: VectorStoreRetriever):
        self.base_retriever = base_retriever

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self.base_retriever.search(query, top_k=5)

        return [
            Document(page_content=r["text"], metadata=r["metadata"])
            for r in results
        ]