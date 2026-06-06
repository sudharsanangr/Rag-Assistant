"""
summarizer.py - Transcript Extraction & LLM Response Generation Module

Purpose:
- Extracts YouTube transcripts from video URLs (Step 2).
- Queries Google Gemini LLM using structured, context-rich prompts (Step 9).
- Formats source timestamps and appends them to answer outputs (Step 10).
- Generates structured, global timeline outlines for entire videos.
"""

import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load API keys and config from the .env environment file
load_dotenv()

# Set the default Google Gemini model for text-only operations
DEFAULT_MODEL = "gemini-2.5-flash"


def extract_video_id(url):
    """
    Extracts the unique 11-character video ID from various YouTube URL formats.
    Matches standard watch URLs, short links, embed links, and mobile links.
    """
    # 1. Regex pattern to capture the 11-char ID after watch?v=, embed/, shorts/, or youtu.be/
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    
    # 2. Fallback: Search for any standalone 11-character ID block
    match = re.search(r'([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    return None


def get_transcript(url):
    """
    Downloads the transcript data for the specified YouTube video.
    
    Args:
        url (str): The full YouTube video link.
        
    Returns:
        list: A list of segment dictionaries, e.g., [{"text": "...", "start": 0.0, "duration": 1.0}]
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    
    # 1. Resolve URL into its unique video ID
    video_id = extract_video_id(url)
    if not video_id:
        print(f"Could not extract YouTube video ID from URL: {url}")
        return []

    # 2. Fetch transcript from the public API
    try:
        api = YouTubeTranscriptApi()
        data = api.fetch(video_id)
        
        # 3. Format and clean raw transcript coordinates into standard float seconds
        return [
            {
                "text": item.text,
                "start": float(item.start),
                "duration": float(item.duration)
            }
            for item in data
        ]
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return []


def format_timestamp(seconds: float) -> str:
    """
    Converts a float duration in seconds into a readable MM:SS or HH:MM:SS string.
    E.g., 65.0 -> '01:05'
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    # Return HH:MM:SS format if the video is longer than an hour, else MM:SS
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def answer_with_context(question, context, timestamp=None, model_name=DEFAULT_MODEL):
    """
    Formulates a strict context-bound prompt, submits it to Gemini, and 
    attaches the source timestamp to the resulting answer.
    
    Args:
        question (str): User query.
        context (str): Text retrieved from the vector database.
        timestamp (float): The exact start time of the best matched context.
        model_name (str): Gemini model selector.
        
    Returns:
        str: Final formatted response.
    """
    # 1. Initialize the Google Gemini chat model
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # 2. Build the system/user prompt enforcing absolute grounding in the transcript
    prompt = f"""
Question:
{question}

Context:
{context}

Answer the user's question.

RULES:
1. Your answer must be based EXCLUSIVELY on the provided transcript context. Do NOT assume, extrapolate, or bring in any outside knowledge.
2. If the context does not contain the answer, you must respond with exactly: "Not mentioned in video"
3. Do not make up or speculate about any information.
4. Keep the answer clear and concise.
5. Structure the response exactly with the following sections:
   - Video Summary:
   - Key Concepts:
   - Important Notes:
   - Study Notes:
   - Question Answer:
   - Follow-up Question:
"""
    
    # 3. Call the Gemini model using LangChain's HumanMessage wrapper
    message = HumanMessage(content=prompt)
    response = llm.invoke([message])
    answer_text = response.content.strip()

    # 4. If the model determines the answer is missing, return fallback message
    if "Not mentioned in video" in answer_text:
        return "Not mentioned in video"

    # 5. Format and append the starting timestamp of the source video section
    formatted_time = format_timestamp(timestamp) if timestamp is not None else "00:00"
    return f"{answer_text}\n\n---\n🎥 **Source Video Section**: `{formatted_time}`"


def extract_video_concepts(transcript, query=None, model_name=DEFAULT_MODEL):
    """
    Generates a structured, chronological timeline and outline of topics covered
    by feeding the entire timed transcript into Gemini.
    
    Args:
        transcript (list): Complete timed transcript list.
        query (str): Optional user prompt to customize the outline focus.
        model_name (str): Gemini model selector.
        
    Returns:
        str: Markdown outline of video chapters/topics.
    """
    # 1. Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # 2. Format the entire transcript into a timed script, e.g., "[01:30] python loops..."
    formatted_lines = []
    for item in transcript:
        time_str = format_timestamp(item["start"])
        formatted_lines.append(f"[{time_str}] {item['text']}")
        
    transcript_text = "\n".join(formatted_lines)

    # 3. Adapt instructions based on whether the user asked a specific outline query
    query_instruction = ""
    if query:
        query_instruction = f"Address the user's specific request: '{query}'."
    else:
        query_instruction = "Provide a comprehensive outline of the topics, events, or concepts covered in the video."

    # 4. Formulate the outline extraction prompt
    prompt = f"""
You are an advanced AI video assistant. Your task is to analyze the video transcript (provided below with timestamps) and generate a comprehensive, structured timeline of topics covered.

CRITICAL REQUIREMENT:
You MUST prefix every single topic, chapter, or concept in your list with its starting timestamp (formatted exactly as `[MM:SS]` or `[HH:MM:SS]`) from the transcript. Do NOT omit timestamps under any circumstances. If you list a concept, it MUST have a timestamp prefix.

INSTRUCTIONS:
1. Scan the full transcript text. Identify all key topics, sections, events, or concepts discussed.
2. Map these sections to their approximate start timestamps in the video.
3. Output a clean, professional markdown outline where EVERY bullet point starts with its mapped timestamp.
4. {query_instruction}
5. Base your response strictly on the provided transcript text. Do not invent topics that are not present.

TRANSCRIPT TEXT WITH TIMESTAMPS:
{transcript_text}

Output format:
## 🎬 Video Timeline & Topics Covered
- **[Timestamp] Section/Topic/Event**: Brief summary of what is discussed or what occurs here.
"""
    # 5. Invoke the Gemini LLM to create the structured markdown timeline
    message = HumanMessage(content=prompt)
    response = llm.invoke([message])
    return response.content.strip()