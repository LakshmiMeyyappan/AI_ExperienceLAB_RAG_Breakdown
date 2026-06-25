from sentence_transformers import SentenceTransformer

def compare_embedding_models():
    """Generates embeddings using different models to compare their outputs."""
    sentences = ["The cat sits outside.", "A dog is playing in the park."]
    
    # List of models to compare
    models_to_test = [
        "all-MiniLM-L6-v2",       # Lightweight, great default
        "BAAI/bge-small-en-v1.5", # High performance for retrieval
        "intfloat/e5-small-v2"    # Excellent semantic search
    ]

    for model_name in models_to_test:
        print(f"\nLoading Model: {model_name}...")
        # Initialize the model (will download the first time you run it)
        model = SentenceTransformer(model_name)
        
        # Generate embeddings
        embeddings = model.encode(sentences)

        print("EMBEDDINGS:", embeddings)
        
        print(f"Embedding Dimension: {embeddings.shape[1]} numbers per sentence")
        print(f"Sample values from first sentence (first 5 dimensions):")
        print(embeddings[0][:5])

if __name__ == "__main__":
    print("--- Module 3: Embedding Generation ---")
    compare_embedding_models()