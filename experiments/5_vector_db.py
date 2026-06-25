import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    print("--- Module 5: FAISS Vector Database ---")
    
    # 1. Prepare data and model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = [
        "Photosynthesis is how plants make food.",
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "Eiffel tower is located in Paris."
    ]
    
    # 2. Generate embeddings
    chunk_embeddings = model.encode(chunks)
    print(f"Chunk Embeddings Shape: {chunk_embeddings.shape}") # Should be (4, 384) for 4 chunks and 384-dim embeddings
    dimension = chunk_embeddings.shape[1] # 384 for MiniLM
    print(f"Embedding Dimension: {dimension} numbers per chunk")
    
    # 3. Build FAISS Index (IndexFlatL2 measures exact Euclidean distance)
    # Note: For exact cosine similarity in FAISS, we usually normalize vectors 
    # and use IndexFlatIP (Inner Product), but L2 distance is perfect for beginners.
    index = faiss.IndexFlatL2(dimension)
    print(f"Is the FAISS index trained? {index.is_trained}") # Should be True for IndexFlatL2
    
    # FAISS requires numpy arrays to be float32
    faiss_data = np.array(chunk_embeddings).astype("float32")
    print(f"Data shape for FAISS: {faiss_data.shape}, Data type: {faiss_data.dtype}")
    print(f"FAISS index initialized with {dimension} dimensions")
    
    index.add(faiss_data) 
    print(f"Total vectors in FAISS index: {index.ntotal}")
    
    # 4. Search
    query = "Tell me about French landmarks."
    query_embedding = model.encode([query]).astype("float32")
    
    k = 2 # Top K results
    distances, indices = index.search(query_embedding, k)
    
    print(f"\nQuery: '{query}'")
    print(f"Top {k} Retrievals:")
    for i in range(k):
        idx = indices[0][i] # Get the original index of the chunk
        dist = distances[0][i] # Get the L2 distance (lower is better for L2)
        print(f"Rank {i+1} | Distance: {dist:.4f} | Chunk: '{chunks[idx]}'")