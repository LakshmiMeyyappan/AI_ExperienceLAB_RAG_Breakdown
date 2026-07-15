import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "mock_vector_db.json")

def _load_vector_db() -> dict:
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def query_vector_store(semantic_query: str) -> str:
    """Simulates a dense vector similarity search across all indexed corporate files."""
    db = _load_vector_db()
    matches = []
    for chunk in db.get("vector_index", []):
        if any(keyword in chunk["text"].lower() for keyword in semantic_query.lower().split()):
            matches.append({"id": chunk["id"], "metadata": chunk["metadata"]})
    return json.dumps(matches) if matches else "Result: No matching vector chunks found."

def filter_by_source_document(target_pdf: str, keyword: str) -> str:
    """Surgically filters and extracts text elements inside a specific verified PDF source."""
    db = _load_vector_db()
    filtered_results = []
    for chunk in db.get("vector_index", []):
        if chunk["metadata"]["source"].lower() == target_pdf.lower():
            if keyword.lower() in chunk["text"].lower():
                filtered_results.append({"id": chunk["id"], "text": chunk["text"]})
    return json.dumps(filtered_results) if filtered_results else f"Result: No matches found inside {target_pdf}."

def fetch_document_chunk(chunk_id: str) -> str:
    """Streams the complete raw text payload of a specific document chunk into context memory."""
    db = _load_vector_db()
    for chunk in db.get("vector_index", []):
        if chunk["id"] == chunk_id:
            return f"Source Content [{chunk['metadata']['source']}]: {chunk['text']}"
    return f"Error: Chunk reference '{chunk_id}' not found in index."

# The Unified Functional Tool Registry Mapping
TOOL_REGISTRY = {
    "query_vector_store": query_vector_store,
    "filter_by_source_document": filter_by_source_document,
    "fetch_document_chunk": fetch_document_chunk
}