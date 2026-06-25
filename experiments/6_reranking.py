from sentence_transformers import CrossEncoder

if __name__ == "__main__":
    print("--- Module 6: Reranking ---")
    
    # CrossEncoders output a single score, not an embedding vector
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    query = "How do I fix a flat tire?"
    
    # Imagine FAISS returned these three. The first one talks about tires but is a joke.
    retrieved_chunks = [
        "I was tired yesterday so my mood was flat.",
        "To fix a flat tire, first use a jack to lift the car, then remove the lug nuts.",
        "You can buy car tires at the local auto shop."
    ]
    
    # We pair the query with each chunk
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    print(f"Pairs for reranking: {pairs}")
    
    # Get scores
    scores = reranker.predict(pairs)
    print(f"Reranking Scores: {scores}")
    
    # Combine chunks and scores, then sort
    print("LIST", list(zip(retrieved_chunks, scores)))


    results = list(zip(retrieved_chunks, scores))
    print(f"Results before sorting: {results}")

    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Query: '{query}'\n")
    print("Reranked Results (Higher score is better):")
    for chunk, score in results:
        print(f"Score: {score:.4f} | Chunk: '{chunk}'")