"""
vector_store.py - RAG Vector Database Module (Text-Only)

Purpose:
- Handles loading and caching of local embedding models.
- Segments raw transcripts into user-friendly time chunks.
- Indexes chunked text into a fast, local in-memory vector database.
- Queries the vector database to perform similarity searches.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Load variables from .env file (like API keys)
load_dotenv()

# Global cache variable so we only load the embedding model once
_embeddings = None


def get_embeddings():
    """
    Loads and caches the Hugging Face 'all-MiniLM-L6-v2' embedding model.
    This model converts text into dense, 384-dimensional numerical vectors.
    """
    global _embeddings
    # If the model is not loaded yet, initialize it
    if _embeddings is None:
        # Step 4: Generate Embeddings using all-MiniLM-L6-v2 model locally
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def create_chunks(transcript, interval_seconds=30):
    """
    Groups raw transcript segments into clean, time-based text chunks (default: 30-second windows).
    This ensures that when a user asks a question, we can retrieve a concise block of 
    text and find its specific timestamp.
    
    Args:
        transcript (list): List of dicts, e.g., [{"text": "hello", "start": 1.2, "duration": 0.5}]
        interval_seconds (int): Desired duration of each chunk in seconds.
        
    Returns:
        list: Chunked data with combined text and start times.
    """
    if not transcript:
        return []

    chunks = []
    current_chunk_text = []
    current_chunk_start = None

    for item in transcript:
        start_time = item["start"]
        text = item["text"]

        # 1. Initialize the start time for the very first chunk
        if current_chunk_start is None:
            current_chunk_start = start_time

        # 2. Check how much time has elapsed since the current chunk started
        time_elapsed = start_time - current_chunk_start

        # 3. If time elapsed exceeds our threshold, finalize the chunk and start a new one
        if time_elapsed >= interval_seconds:
            if current_chunk_text:
                chunks.append({
                    "text": " ".join(current_chunk_text).strip(),
                    "start": float(current_chunk_start)
                })
            # Reset chunk tracking for the new interval
            current_chunk_start = start_time
            current_chunk_text = [text]
        else:
            # 4. Otherwise, continue appending text to the current chunk
            current_chunk_text.append(text)

    # 5. Append any remaining text as the final chunk
    if current_chunk_text:
        chunks.append({
            "text": " ".join(current_chunk_text).strip(),
            "start": float(current_chunk_start)
        })

    return chunks


def create_vector_db(transcript):
    """
    Segments the transcript, runs local embeddings, and stores them in an in-memory database.
    
    Args:
        transcript (list): Raw transcript list from youtube-transcript-api.
        
    Returns:
        InMemoryVectorStore: An initialized database loaded with transcript chunks.
    """
    # Step 3: Segment transcript into 30-second intervals
    chunks = create_chunks(transcript, interval_seconds=30)

    # Convert chunks into LangChain's Document object layout (stores text and metadata start times)
    documents = [
        Document(
            page_content=c["text"],
            metadata={"start": c["start"]}
        )
        for c in chunks
    ]

    # Step 5: Store vectors in a fast, in-memory local vector store
    db = InMemoryVectorStore(get_embeddings())
    db.add_documents(documents)

    return db


def search_query(question, db, k=5):
    """
    Queries the vector store for the top-k most semantically similar text chunks.
    
    Args:
        question (str): User query/question.
        db (InMemoryVectorStore): The active vector database.
        k (int): Number of matching chunks to retrieve.
        
    Returns:
        list: Top matching chunks sorted by relevance score.
    """
    if db is None:
        return []

    # Step 7: Perform similarity search returning the matching documents and their distance scores
    results = db.similarity_search_with_score(question, k=k)

    docs = []
    for doc, score in results:
        meta = doc.metadata or {}
        # Append the chunk details, start time, and similarity score
        docs.append({
            "text": doc.page_content,
            "start": meta.get("start", 0.0),
            "score": float(score)
        })

    # Sort results by score (lowest distance score means highest match)
    docs.sort(key=lambda x: x["score"])
    return docs


def get_best_timestamp(docs):
    """
    Extracts the starting timestamp from the highest matching retrieved document.
    """
    if not docs:
        return 0.0
    return float(docs[0].get("start", 0.0))