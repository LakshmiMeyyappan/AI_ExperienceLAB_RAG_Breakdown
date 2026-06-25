Markdown
# Core RAG Breakdown

This is a minimal, beginner-friendly implementation of Retrieval-Augmented Generation (RAG) built from first principles.

**Goal:** Understand how RAG works by running small, clear demos that each isolate and show exactly one core engineering concept.

---

## 📂 What You Get (The Demos)

### 📈 Week 1 — Foundational Two-Stage Retrieval
* **`01 — Chunking (Fixed & Overlap)`** — Demonstrates document splitting strategies to prevent data loss.
* **`02 — Embedding Models Comparison`** — Compares different embedding models and prints core vector shapes and metrics.
* **`03 — Cosine Similarity`** — Computes mathematical similarity scores between short sentences and maps out a matrix.
* **`04 — Vector Search (No FAISS)`** — A brute-force, pure-Python embedding-based document search loop.
* **`05 — Top-K Retrieval`** — Filters and pulls only the top-K relevant passages to minimize prompt bloat.
* **`06 — FAISS Vector DB`** — Demonstrates lightning-fast nearest-neighbor lookups using an in-memory FAISS index.
* **`07 — Complete Core RAG Pipeline`** — Combines everything into an end-to-end flow: Chunk → Embed → Retrieve → Mock LLM Answer.

### 🛡️ Week 2 — Enterprise Advanced Orchestration
* **`08 — Advanced RAG Orchestrator`** — A unified production-grade script that executes a local loop showcasing:
  * **HyDE Optimization:** Generates predictive "search bait" responses to fix conversational query gaps.
  * **Parent-Child Hydration:** Searches tiny sentence-level chunks and auto-swaps them for complete parent context files.
  * **Context Token Compression:** Evaluates system limits and automatically shrinks context windows to protect model budgets.
  * **Deterministic JSON Generation:** Restricts the model to raw structured data outputs with no chatbot filler.
  * **RAG Triad Metrics:** Runs automated algorithmic self-evaluation matrix scoring for safety and accuracy.


## 🛠️ Quick Setup

### 1. Initialize Your Environment
Create and activate your Python virtualization space first:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
# or use: .venv\Scripts\activate (cmd)

python -m pip install -r requirements.txt
2. Start Your Local LLM Engine (For Week 2)
Ensure you have Ollama running on your machine, then pull the optimized model locally:

PowerShell
ollama run qwen2.5:1.5b
💻 Running the Experiments
Copy and run these directly in your terminal to see the pipelines execute:

PowerShell
# Run Week 1 Foundations
python RAG-Breakdown\experiments\01_Chunking_Embeddings.py
python RAG-Breakdown\experiments\02_embedding_models.py
python RAG-Breakdown\experiments\03_cosine_similarity.py
python RAG-Breakdown\experiments\04_vector_search.py
python RAG-Breakdown\experiments\05_top_k_retrieval.py
python RAG-Breakdown\experiments\06_faiss_vector_db.py
python RAG-Breakdown\experiments\07_complete_rag_pipeline.py

# Run Week 2 Production Orchestration
python RAG-Breakdown\experiments\08_advanced_rag_orchestrator.py

📊 Sample Output Snippets
What to expect in your console terminal (perfect for screenshots):

Plaintext
01 — Chunking (Fixed & Overlap)
------------------------------------------------------------
Document 1: ai
Fixed chunk 1 — 200 chars
Artificial Intelligence (AI) is a field of computer science focused on creating systems that can perform tasks...

02 — Embedding Models Comparison
------------------------------------------------------------
Model: MiniLM (all-MiniLM-L6-v2)
Embedding Dimension: 384
First Vector Values: [0.0021, -0.0013, ...]
Observation: Compact and fast, suitable for lower-memory retrieval.

08 — Enterprise Advanced RAG Orchestrator (Ollama Local)
------------------------------------------------------------
Target Query: 'Can you check what kind of backend architectural logic InvoiceIQ runs on?'
 ↳ HyDE Document Target : "InvoiceIQ uses a combination of microservices architecture..."
 ↳ Matched Child 'c1' -> Hydrated Parent 'doc_1'
 ↳ Words BEFORE Compression: 51 -> AFTER Compression: 26 (Budget Managed)

Production Output JSON Dictionary:
{
    "answer": "InvoiceIQ runs on a FastAPI backend.",
    "confidence_score": 1.0,
    "source_found": true
}

==================== FINAL SYSTEM RUN DIAGNOSTICS ====================
 Context Relevance Score       : 25.0%
 Groundedness Safety Rating    : 18.5%
 Output Answer Alignment Score : 25.0%
======================================================================
Total system transaction execution time: 22.7 seconds