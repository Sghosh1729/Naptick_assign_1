import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from preprocessing import all_documents  # You should return both `texts` and `collection_names`

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load Hugging Face embedding model

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Mean Pooling with attention mask

'''def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)
'''
# Embed a list of strings

def get_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True)

# Prepare data and metadata

texts = [doc.page_content for doc in all_documents if doc.page_content]
metadata = [{"collection_name": doc.metadata.get("collection_name", "Unknown")} for doc in all_documents if doc.page_content]
embeddings = get_embeddings(texts)

# Create FAISS index and metadata store

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save metadata in parallel
np.save("metadata.npy", metadata)
faiss.write_index(index, "semantic_index.faiss")

print(f" FAISS index saved with {len(texts)} items.")


def load_index_and_metadata(index_path="semantic_index.faiss", metadata_path="metadata.npy"):
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Index or metadata file not found. Please build the index first.")

    index = faiss.read_index(index_path)
    metadata = np.load(metadata_path, allow_pickle=True)
    return index, metadata.tolist()


def search_with_metadata(query, top_k=3, collection_filter=None):
    """
    Perform semantic search with metadata filtering.

    :param query: Query string
    :param top_k: Number of top results to return
    :param collection_filter: Optional metadata filter (e.g., collection name)
    :return: List of filtered results
    """
    query_embedding = get_embeddings([query])
    _, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        doc_metadata = metadata[idx]

        # Apply collection filter if specified
        if collection_filter and doc_metadata["collection_name"] != collection_filter:
            continue  # Skip documents that don't match the filter

        results.append({
            "text": texts[idx],
            "collection": doc_metadata["collection_name"]
        })

    return results


# Example usage:
query = "What health data is available?"
collection_filter = "wearable"  # Optional: You can filter by a specific collection
results = search_with_metadata(query, top_k=3, collection_filter=collection_filter)

# Print results
print("Top results:")
print(results)
for result in results:
    print(f"\n[From: {result['collection']}]")
    print(result['text'])
