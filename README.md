# 🎥 YouTube Video RAG Assistant

## Overview

YouTube Video RAG Assistant is an AI-powered application that enables users to ask questions about any YouTube video and receive accurate answers based on the video's transcript.

The system uses Retrieval-Augmented Generation (RAG) to retrieve the most relevant transcript sections and generate context-aware responses using Google Gemini. It also provides the timestamp of the video segment from which the answer was derived.

---

## Features

* Extracts transcripts from YouTube videos
* Creates time-based transcript chunks
* Generates semantic embeddings using Hugging Face models
* Stores embeddings in a Vector Database
* Performs similarity search to retrieve relevant content
* Answers user questions using Google Gemini
* Returns source timestamps for answer traceability
* Supports interactive question-answering on video content

---

## Tech Stack

* Python
* Google Gemini API
* LangChain
* Hugging Face Embeddings (all-MiniLM-L6-v2)
* Vector Database
* YouTube Transcript API

---

## Project Workflow

1. User provides a YouTube video URL.
2. Transcript is extracted with timestamps.
3. Transcript is divided into time-based chunks.
4. Embeddings are generated for each chunk.
5. Embeddings are stored in a Vector Database.
6. User asks a question about the video.
7. Relevant transcript chunks are retrieved using semantic search.
8. Retrieved context is sent to Gemini.
9. Gemini generates an answer based on the retrieved content.
10. The corresponding video timestamp is returned.

---

## Example Output

**Question**

What is a while loop?

**Answer**

A while loop is a programming construct that repeatedly executes a block of code while a specified condition remains true.

**Timestamp**

42:23

---

## Project Structure

```text
Video_Rag/
│
├── app.py
├── summarizer.py
├── vector_store.py
├── requirements.txt
├── .env
└── README.md
```

---

## Installation

```bash
git clone <repository-url>

cd Video_Rag

pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Run the Project

```bash
python app.py
```

Enter a YouTube URL and start asking questions about the video.

---

## Key Learning Outcomes

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* Embedding Models
* Prompt Engineering
* LLM Integration
* LangChain Workflows

---

## Author

Developed as a practical RAG application to explore semantic search, vector databases, and Large Language Model integration using real-world YouTube content.
