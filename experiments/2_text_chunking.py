def chunk_text(text, chunk_size=100, overlap=20):
    """
    Splits text into chunks of `chunk_size` characters with an `overlap`.
    Note: For production, we usually chunk by words or tokens, but character 
    chunking is easiest to understand for beginners.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Get the slice of text
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move the start pointer forward, subtracting the overlap
        # If overlap is 20, we step back 20 characters to start the next chunk
        start = start + chunk_size - overlap

    return chunks

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" ✂️  VISUALIZING TEXT CHUNKING MECHANICS (CHARACTER-LEVEL) ")
    print("="*70)
    
    sample_text = "Retrieval-Augmented Generation (RAG) is a technique that bridges the gap between large language models and external knowledge bases. It allows the model to look up facts before answering."
    text_length = len(sample_text)
    print(f"\n📝 ORIGINAL TEXT (Total Length: {text_length} characters):")
    print(f"   \"{sample_text}\"")
    print("-" * 70)
    
    # ------------------------------------------------------------------
    # CONFIGURATION 1: FIXED-SIZE CHUNKING (NO OVERLAP)
    # ------------------------------------------------------------------
    size_fixed = 50
    overlap_fixed = 0
    fixed_chunks = chunk_text(sample_text, chunk_size=size_fixed, overlap=overlap_fixed)
    
    print(f"\n[CONFIG 1] Fixed-Size Chunking (Size: {size_fixed} chars, Overlap: {overlap_fixed} chars)")
    print(f"📊 Generated {len(fixed_chunks)} unique chunks:")
    
    for i, c in enumerate(fixed_chunks):
        # Calculate indices based on step size for fixed chunking
        start_idx = i * size_fixed
        end_idx = min(start_idx + size_fixed, text_length)
        print(f"  🧩 Chunk {i+1} [{start_idx:03d}:{end_idx:03d}] -> '{c}'")

    print("-" * 70)

    # ------------------------------------------------------------------
    # CONFIGURATION 2: OVERLAPPING CHUNKING
    # ------------------------------------------------------------------
    size_overlap = 50
    overlap_val = 15
    overlap_chunks = chunk_text(sample_text, chunk_size=size_overlap, overlap=overlap_val)
    
    print(f"\n[CONFIG 2] Overlapping Chunking (Size: {size_overlap} chars, Overlap: {overlap_val} chars)")
    print(f"📊 Generated {len(overlap_chunks)} overlapping chunks:")
    
    for i, c in enumerate(overlap_chunks):
        # Calculate starting index based on the sliding step size: (size - overlap)
        step = size_overlap - overlap_val
        start_idx = i * step
        end_idx = min(start_idx + size_overlap, text_length)
        
        print(f"\n  🧩 Chunk {i+1} [{start_idx:03d}:{end_idx:03d}] -> '{c}'")
        
        # Look ahead to print what text is being repeated in the overlap
        if i < len(overlap_chunks) - 1:
            next_start = (i + 1) * step
            # Overlap text exists between the next chunk's start and this chunk's end
            overlap_text = sample_text[next_start:end_idx]
            if overlap_text:
                print(f"     └─ 🔄 OVERLAP WITH CHUNK {i+2}: '{overlap_text}'")

    print("\n" + "="*70)