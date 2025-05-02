import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocessing import all_documents

class VectorStoreRetriever:
    def __init__(self, index_path="semantic_index.faiss", metadata_path="metadata.npy"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None
        self.texts = []
        self.metadata = []

    def build_index(self, documents):
        """
        Build and save FAISS index from a list of documents.

        :param documents: List of Document objects with `page_content` and `metadata`.
        """
        self.texts = [doc.page_content for doc in documents if doc.page_content]

        self.metadata = []
        for doc in documents:
            meta = doc.metadata or {}
            collection = meta.get("source", "unknown")
            self.metadata.append({"collection_name": collection.lower()})

        # Compute embeddings
        embeddings = self.model.encode(self.texts, convert_to_numpy=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Save index and metadata
        faiss.write_index(self.index, self.index_path)
        np.save(self.metadata_path, self.metadata)
        np.save("texts.npy", self.texts)

        #print(f"FAISS index built with {len(self.texts)} documents.")

    def load_index(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path) or not os.path.exists(
                "texts.npy"):
            raise FileNotFoundError("Index, metadata, or texts file not found. Run `build_index()` first.")

        self.index = faiss.read_index(self.index_path)
        self.metadata = np.load(self.metadata_path, allow_pickle=True).tolist()
        self.texts = np.load("texts.npy", allow_pickle=True).tolist()
        #print(f" FAISS index loaded with {len(self.metadata)} documents.")
    def search(self, query, top_k=5, collection_filter=None, return_scores=False):
        """Semantic search with optional collection filtering."""
        if self.index is None or not self.metadata:
            self.load_index()

        # Normalize collection_filter to lowercase list
        if isinstance(collection_filter, str):
            collection_filter = [collection_filter]
        if collection_filter:
            collection_filter = [c.lower() for c in collection_filter]
        #print(collection_filter)
        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.metadata):
                print("1")
                continue

            meta = self.metadata[idx]
            collection = meta.get("collection_name", "unknown").lower()
            #print(collection)
            # Filter by collection
            if collection_filter and collection not in collection_filter:
                continue

            result = {
                "text": self.texts[idx],
                "collection": collection,
                "metadata": meta
            }
            if return_scores:
                result["score"] = float(distances[0][i])

            results.append(result)

        return results

# from vector_store import VectorStoreRetriever
retriever = VectorStoreRetriever()

# Build index (once)
retriever.build_index(all_documents)  # where all_documents = list of LangChain-like Document objects

# Search
query = "How much did I sleep last night?"
results = retriever.search(query, top_k=3, collection_filter="wearable", return_scores=True)
#print(all_documents)
#print(results)
'''for r in results:
    print(f"[{r['collection']}] {r['text']} (score: {r['score']:.2f})")'''


