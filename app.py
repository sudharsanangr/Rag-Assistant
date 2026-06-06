"""
app.py - Command Line Interface (CLI) YouTube Video RAG Assistant

Purpose:
- Provides a CLI alternative for testing the text-only RAG pipeline.
- Performs URL input, transcript download, vector DB chunking, and semantic search.
- Queries Gemini and prints formatted answers with source timestamps.
"""

import re
from difflib import SequenceMatcher
import os

# Block 1: Import core RAG functions from summarizer and vector_store
from summarizer import (
    get_transcript,
    answer_with_context,
    extract_video_concepts
)

from vector_store import (
    create_vector_db,
    search_query
)


def choose_precise_timestamp(question, transcript, base_start, window=10):
    """
    Refines the coarse chunk-level timestamp down to the exact second where
    the matching phrase or word overlap is spoken in the video.
    
    Args:
        question (str): The user's search query.
        transcript (list): Complete timed transcript segment list.
        base_start (float): The start time of the coarse matching chunk (e.g., 30s bucket).
        window (int): The search window in seconds around the base start time.
    """
    if not transcript:
        return base_start

    # Convert question to lowercase and extract words for keyword matching
    question_text = question.lower()
    question_words = set(re.findall(r"\w+", question_text))
    
    best_timestamp = base_start
    best_score = -1.0

    # Loop through all raw transcript sentences to find the best match within the time window
    for item in transcript:
        # Filter: Skip sentences spoken outside our time window
        if abs(item["start"] - base_start) > window:
            continue

        text = item["text"].lower()
        if not text:
            continue

        # Calculate similarity score: combining string similarity and keyword matching
        similarity = SequenceMatcher(None, text, question_text).ratio()
        word_overlap = sum(1 for w in question_words if w and w in text)
        combined_score = similarity + word_overlap * 0.15

        # Update if we find a closer match
        if combined_score > best_score:
            best_score = combined_score
            best_timestamp = item["start"]

    return best_timestamp


# ============================================================================
# MAIN EXECUTION LOOP
# ============================================================================

print("\n=== YouTube AI Assistant (CLI Version) ===")

# 1. Ask user for the YouTube link
url = input("\nEnter YouTube URL: ")

# 2. Download transcript data (Step 2)
print("Fetching transcript...")
transcript = get_transcript(url)

if not transcript:
    print("No transcript found or captions are disabled.")
    exit()

# 3. Create the vector database locally (Steps 3-5)
print("Segmenting transcript and creating local vector store...")
db = create_vector_db(transcript)

print("\nReady! Ask questions about the video. Type 'exit' or 'stop' to quit.")

# 4. Interactive chat loop (Steps 6-10)
while True:
    # Get question from terminal input
    q = input("\nAsk: ")

    # Stop the program if the user enters a stop command
    if q.lower() in ["exit", "stop"]:
        break

    # Check if the query is a global summary/outline request
    is_global = any(kw in q.lower() for kw in [
        "concept", "syllabus", "topics", "summary", "summarize", 
        "overview", "outline", "chapters", "agenda", "content covered", 
        "all the things taught", "what is this video about", "what does this video cover"
    ])

    if is_global:
        # Global RAG: Generate outline using the entire transcript
        print("\nAnalyzing entire transcript for timeline outline...")
        print(extract_video_concepts(transcript, query=q))
        continue

    # Standard RAG: Retrieve top matching chunks from local Vector DB
    docs = search_query(q, db)

    if not docs:
        print("Not mentioned in video")
        continue

    # Combine text from matching documents to build context
    context = "\n".join([d["text"] for d in docs[:3]])
    
    # Get coarse chunk start time and refine it to precise timestamp
    coarse_timestamp = docs[0]["start"]
    precise_timestamp = choose_precise_timestamp(q, transcript, coarse_timestamp)

    # Generate answer using Gemini and print results
    answer = answer_with_context(
        q,
        context,
        timestamp=precise_timestamp
    )

    print("\n--- ANSWER ---")
    print(answer)
    print("\nRefined Timestamp:", precise_timestamp)