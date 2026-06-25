import os

def load_documents(data_folder):
    """Loads all .txt files from a specified folder."""
    documents = []
    
    # Check if the folder exists
    if not os.path.exists(data_folder):
        print(f"Folder '{data_folder}' not found. Please create it and add .txt files.")
        return documents

    # Loop through all files in the directory
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                # Store a dictionary with metadata (the filename) and the actual text
                documents.append({"filename": filename, "text": content})
                print(f"Loaded: {filename} ({len(content)} characters)")
                
    return documents

if __name__ == "__main__":
    print("--- Module 1: Document Loading ---")
    docs = load_documents("./data")
    if docs:
        print(f"\nTotal documents loaded: {len(docs)}")
        print(f"Sample content from first doc: {docs[0]['text'][:50]}...")