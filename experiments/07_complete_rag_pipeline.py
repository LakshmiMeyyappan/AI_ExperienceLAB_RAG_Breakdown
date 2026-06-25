import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# ------------------------------------------------------------------
# PATH MANAGEMENT & AUTO-DATA GENERATION (Defensive Engineering)
# ------------------------------------------------------------------
# Dynamically locate the folder where this script actually lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Your exact text datasets stored as defaults to prevent execution crashes
DEFAULT_DATASETS = {
    "ai.txt": """Artificial Intelligence (AI) is a field of computer science focused on creating systems that can perform tasks that typically require human intelligence.
Machine Learning (ML) is a subset of AI where models learn patterns from data rather than following explicitly programmed rules.
Deep Learning is a specialized area of Machine Learning that uses neural networks with multiple layers to learn complex representations.
Neural Networks are inspired by the human brain and consist of interconnected nodes called neurons.
Natural Language Processing (NLP) enables machines to understand, interpret, and generate human language.
Computer Vision allows machines to analyze and understand images and videos.
Large Language Models (LLMs) such as GPT are trained on vast amounts of text data and can perform tasks such as question answering, summarization, and content generation.
AI applications include healthcare diagnosis, recommendation systems, autonomous vehicles, fraud detection, and virtual assistants.
Training AI models requires large datasets, computational resources, and careful evaluation.
AI systems must be monitored to reduce bias, hallucinations, and inaccurate outputs.""",

    "python.txt": """Python is one of the most widely used programming languages for Artificial Intelligence and Data Science.
Python is known for its readability, simplicity, and extensive ecosystem of libraries.
NumPy provides efficient numerical computations using multidimensional arrays.
Pandas is used for data manipulation, cleaning, and analysis.
Matplotlib and Plotly are commonly used for data visualization.
Scikit-learn provides tools for machine learning algorithms such as regression, classification, and clustering.
TensorFlow and PyTorch are popular deep learning frameworks.
FastAPI is commonly used for deploying machine learning and AI applications as APIs.
Python supports integration with cloud platforms, databases, and external services.
Python has become the primary language for AI engineering because of its flexibility and strong community support.""",

    "cloud.txt": """Cloud computing provides on-demand access to computing resources over the internet.
Amazon Web Services (AWS) is one of the leading cloud platforms.
Amazon EC2 provides virtual machines that allow users to deploy applications.
Amazon S3 provides scalable object storage for files and datasets.
AWS Lambda enables serverless execution of code without managing infrastructure.
Docker allows applications to be packaged into portable containers.
Kubernetes helps manage and scale containerized applications.
Many AI systems use cloud infrastructure to host models and process large amounts of data.
Monitoring and observability are important for production AI systems to track performance, latency, and failures.
Cloud platforms help organizations deploy reliable and scalable AI applications."""
}

def ensure_knowledge_base_exists():
    """Validates paths and automatically self-heals empty environments."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created target directory: {DATA_DIR}")
        
    for filename, text_content in DEFAULT_DATASETS.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text_content.strip())
            print(f"📝 Auto-generated knowledge file: {filename}")

def clean_and_load_text(filepath):
    """Loads a file and splits it into clean, isolated lines/sentences."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

def create_overlapping_chunks(lines, lines_per_chunk=2, overlap_lines=1):
    """
    Groups sentences into logical, text-preserved semantic blocks.
    Prevents words from being cut in half.
    """
    chunks = []
    i = 0
    while i < len(lines):
        chunk_slice = lines[i:i + lines_per_chunk]
        chunk_text = " ".join(chunk_slice)
        chunks.append(chunk_text)
        i += (lines_per_chunk - overlap_lines)
    return chunks

# ------------------------------------------------------------------
# MAIN EXECUTION RUNTIME
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" 🛠️  EXPLORING THE CORE RAG RETRIEVAL PIPELINE STEP-BY-STEP ")
    print("="*70)
    
    # Run absolute path data verification
    ensure_knowledge_base_exists()
    
    # 1. INITIALIZE TRANSFORMERS
    print("\n[STEP 1] Initializing Specialized Embedding & Reranking Models...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("✔️ Models loaded successfully.")

    # 2. DOCUMENT LOADING & CHUNKING
    print("\n[STEP 2] Loading Knowledge Files & Processing Overlapping Chunks...")
    target_files = {
        "AI Concepts": os.path.join(DATA_DIR, "ai.txt"),
        "Python Ecosystem": os.path.join(DATA_DIR, "python.txt"),
        "Cloud Infrastructure": os.path.join(DATA_DIR, "cloud.txt")
    }
    
    all_chunks = []
    for doc_name, path in target_files.items():
        lines = clean_and_load_text(path)
        if not lines:
            continue
        doc_chunks = create_overlapping_chunks(lines, lines_per_chunk=2, overlap_lines=1)
        all_chunks.extend(doc_chunks)
        print(f" 📂 Loaded '{doc_name}': Split {len(lines)} raw lines into {len(doc_chunks)} chunks.")

    # Safety Guard Clause to satisfy rigorous production standards
    if not all_chunks:
        print("\n❌ Critical Failure: No text chunks could be generated. Halting execution.")
        sys.exit(1)

    print(f"\n📊 Total Chunks Registered in Memory System: {len(all_chunks)}")
    print("-" * 70)
    print("🔍 PREVIEW OF THE FIRST 3 GENERATED OVERLAPPING CHUNKS:")
    for idx, sample in enumerate(all_chunks[:3]):
        print(f"  Chunk [{idx}]: \"{sample}\"")
    print("-" * 70)

    # 3. GENERATING HIGH-DIMENSIONAL EMBEDDINGS
    print("\n[STEP 3] Computing Dense Vector Embeddings...")
    embeddings = embedder.encode(all_chunks).astype("float32")
    print(f"✔️ Successfully generated matrix of shape: {embeddings.shape}")
    print(f"💡 Explanation: {embeddings.shape[0]} chunks converted into numerical vectors of dimension {embeddings.shape[1]}.")

    # 4. INDEXING INSIDE FAISS (VECTOR DATABASE)
    print("\n[STEP 4] Constructing and Populating FAISS Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  
    index.add(embeddings)
    print(f"✔️ FAISS Database synchronized. Total indexed vectors: {index.ntotal}")

    # 5. USER QUERY & RETRIEVAL
    query = "Which language is heavily used for Artificial Intelligence and Data Science?"
    print(f"\n🎯 USER QUERY SUBMITTED: \"{query}\"")
    
    print("\n[STEP 5] Stage 1 Retrieval: Fetching Top 4 Chunks via FAISS Vector Match...")
    query_vector = embedder.encode([query]).astype("float32")
    
    k_retrieve = 4 
    distances, indices = index.search(query_vector, k_retrieve)
    
    retrieved_chunks = []
    print("\n💡 STAGE 1 RAW RESULTS (Lower Distance = Closer Vector Match):")
    print(f" {'Rank':<6} | {'FAISS Index':<12} | {'L2 Distance Score':<18} | {'Chunk Text Content Preview'}")
    print(" " + "-"*90)
    
    for rank_idx in range(k_retrieve):
        chunk_idx = indices[0][rank_idx]
        distance_score = distances[0][rank_idx]
        chunk_text = all_chunks[chunk_idx]
        retrieved_chunks.append(chunk_text)
        
        preview = chunk_text if len(chunk_text) < 55 else chunk_text[:52] + "..."
        print(f" #{rank_idx+1:<4} | ID: {chunk_idx:<8} | {distance_score:<18.4f} | \"{preview}\"")

    # 6. CROSS-ENCODER PRECISION RERANKING
    print("\n[STEP 6] Stage 2 Precision Reranking via Cross-Encoder...")
    print("💡 Mechanism: Analyzing query and retrieved chunks simultaneously for exact linguistic alignment.")
    
    query_chunk_pairs = [[query, chunk] for chunk in retrieved_chunks]
    rerank_scores = reranker.predict(query_chunk_pairs)
    
    reranked_results = list(zip(retrieved_chunks, rerank_scores))
    reranked_results.sort(key=lambda x: x[1], reverse=True)

    print("\n💡 STAGE 2 RERANKED RESULTS (Higher Logit Score = Stronger Relevance):")
    print(f" {'Rank':<6} | {'Cross-Encoder Score':<20} | {'Chunk Text Content Preview'}")
    print(" " + "-"*90)
    for rank_idx, (chunk_text, score) in enumerate(reranked_results):
        preview = chunk_text if len(chunk_text) < 55 else chunk_text[:52] + "..."
        print(f" #{rank_idx+1:<4} | {score:<20.4f} | \"{preview}\"")

    # FINAL PRODUCTION COMPOSITION
    print("\n" + "="*70)
    print(" 🌟 FINAL AGGREGATED RETRIEVED CONTEXT FOR THE LLM PROMPT ")
    print("="*70)
    print("Below are the top 2 highly optimal context blocks selected to answer the user request:\n")
    
    for final_rank in range(2):
        chunk, final_score = reranked_results[final_rank]
        print(f"📌 [CONTEXT BLOCK {final_rank+1}] (Relevance Score: {final_score:.2f})")
        print(f"   \"{chunk}\"\n")
        
    print("="*70)
    print("🚀 Pipeline complete. This clean payload is ready for LLM prompt context injection.")