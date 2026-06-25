import numpy as np
from sentence_transformers import SentenceTransformer

def calculate_cosine_similarity(vec_a, vec_b):
    """Calculates the cosine similarity between two vectors using plain NumPy."""
    dot_product = np.dot(vec_a, vec_b)
    print(f"Dot Product: {dot_product}")
    norm_a = np.linalg.norm(vec_a)
    print(f"Norm of Vector A: {norm_a}")
    norm_b = np.linalg.norm(vec_b)
    print(f"Norm of Vector B: {norm_b}")
    return dot_product / (norm_a * norm_b)

if __name__ == "__main__":
    print("--- Module 4: Cosine Similarity ---")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    query = "Where is the feline?"
    documents = [
        "The cat sits on the mat.",
        "I love eating pizza.",
        "Dogs are great pets."
    ]
    
    # 1. Embed the query and the documents
    query_embedding = model.encode(query)
    print(f"Query Embedding: {query_embedding}")
    doc_embeddings = model.encode(documents)
    print(f"Document Embeddings: {doc_embeddings}")
    
    # 2. Calculate similarities
    results = []
    for i, doc_emb in enumerate(doc_embeddings):
        score = calculate_cosine_similarity(query_embedding, doc_emb)
        print(f"Cosine Similarity with Document {i}: {score:.4f}")
        results.append((documents[i], score))
        
    # 3. Rank them based on score (highest to lowest)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Query: '{query}'\n")
    print("Ranking results:")
    for doc, score in results:
        print(f"Score: {score:.4f} | Document: '{doc}'")