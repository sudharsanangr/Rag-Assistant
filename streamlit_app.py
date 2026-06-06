"""
streamlit_app.py - Main Streamlit Frontend Application

Purpose:
- Renders the web interface for the YouTube RAG assistant.
- Manages user session state for chat history, video processing state, and database instances.
- Coordinates the RAG pipeline: URL input -> Transcript download -> Embedding creation -> Chat UI query -> Response generation.
"""

import os
# Block 1: Suppress Streamlit's file watcher.
# This prevents Streamlit from scanning external libraries (like 'transformers') in the venv,
# which otherwise triggers lazy-loading imports and prints errors to the console.
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import re
from datetime import datetime
from difflib import SequenceMatcher

# Block 2: Import local helper functions from our custom scripts
from summarizer import get_transcript, answer_with_context, extract_video_concepts
from vector_store import create_vector_db, search_query


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
# SESSION STATE INITIALIZATION
# ============================================================================
# These keys store user-specific data that persists across page refreshes.

# 1. Stores chat history list
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Stores the currently loaded YouTube URL
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = ""

# 3. Boolean flag indicating if the database is active and ready
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

# 4. Stores the full raw transcript
if "transcript" not in st.session_state:
    st.session_state.transcript = []

# 5. Stores the current in-memory vector store database
if "db" not in st.session_state:
    st.session_state.db = None

# 6. A counter key used to force-refresh the text area widget to clear it
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0

# 7. Tracks whether to display the "RAG Assistant Ready" message after a rerun
if "show_ready_message" not in st.session_state:
    st.session_state.show_ready_message = False


# ============================================================================
# PAGE CONFIGURATION & AESTHETICS
# ============================================================================
st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium CSS styles (harmonious colors, clean borders, glassmorphism card layouts)
st.markdown("""
    <style>
    .main {
        padding: 1.5rem;
        background-color: #fafbfc;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
    }
    .chat-message {
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        font-size: 0.975rem;
        line-height: 1.5;
        transition: transform 0.2s ease;
    }
    .chat-message:hover {
        transform: translateY(-2px);
    }
    .user-message {
        background-color: #e0f2fe;
        border-left: 5px solid #0284c7;
        color: #0f172a;
    }
    .assistant-message {
        background-color: #f8fafc;
        border-left: 5px solid #10b981;
        color: #0f172a;
        border: 1px solid #e2e8f0;
    }
    .gradient-text {
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR - CONTROL PANEL & SETTINGS
# ============================================================================
with st.sidebar:
    st.title("⚙️ RAG Control Panel")
    st.write("Configure your YouTube Video RAG settings here.")
    st.write("---")
    
    # 1. URL Input Block
    youtube_url_input = st.text_input(
        "Paste YouTube URL:",
        value=st.session_state.youtube_url if st.session_state.youtube_url else "",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a full YouTube URL to begin extraction and RAG pipeline setup."
    )
    
    st.write("---")
    st.subheader("Model Configuration")
    
    # 2. Selectbox to choose which Gemini model variant to query
    model_name = st.selectbox(
        "Select Model:",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-pro", "gemini-2.5-pro"],
        help="Choose the underlying Google Gemini model for response generation."
    )
    
    # 3. Temperature slider control
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Controls creativity. Lower is more precise and strict."
    )
    
    st.write("---")
    st.subheader("Action Center")
    
    # 4. Trigger button to initialize processing (Sequential, un-nested logic)
    if st.button("Process Video ⚡", use_container_width=True):
        if youtube_url_input:
            # Clear old database/chat history if the URL has changed
            if youtube_url_input != st.session_state.youtube_url:
                st.session_state.messages = []
                st.session_state.video_processed = False
                st.session_state.transcript = []
                st.session_state.db = None
                st.session_state.show_ready_message = False
                st.session_state.youtube_url = youtube_url_input
            
            # Step A: Download Transcript
            transcript = []
            with st.spinner("Extracting YouTube Transcript..."):
                try:
                    transcript = get_transcript(youtube_url_input)
                except Exception as e:
                    st.error(f"Error fetching transcript: {str(e)}")
            
            # Step B: Segment and Index in local Vector DB
            if not transcript:
                st.error("❌ Transcript extraction failed. Make sure public subtitles are enabled on the video.")
            else:
                st.session_state.transcript = transcript
                
                db = None
                with st.spinner("Chunking & embedding transcript (all-MiniLM-L6-v2)..."):
                    try:
                        db = create_vector_db(transcript)
                        st.session_state.db = db
                    except Exception as e:
                        st.error(f"Error initializing RAG database: {str(e)}")
                
                # Step C: Finalize ready state and trigger a rerun to brighten the screen instantly
                if db is not None:
                    st.session_state.video_processed = True
                    st.session_state.show_ready_message = True
                    st.rerun()
        else:
            st.warning("Please specify a YouTube video URL.")
            
    # 5. Session reset button
    if st.button("Reset Session 🔄", use_container_width=True):
        st.session_state.messages = []
        st.session_state.youtube_url = ""
        st.session_state.video_processed = False
        st.session_state.transcript = []
        st.session_state.db = None
        st.session_state.show_ready_message = False
        st.rerun()

    # 6. Display persistent status message if ready (outside button click block)
    # This renders only after the page reruns and the screen brightens/is ready.
    if st.session_state.get("show_ready_message", False):
        st.write("---")
        st.success("🎉 RAG Assistant Ready!")


# ============================================================================
# MAIN WINDOW LAYOUT (SPLIT PANEL VIEW)
# ============================================================================
st.markdown("<h1>🎥 YouTube Video <span class='gradient-text'>RAG Assistant</span></h1>", unsafe_allow_html=True)
st.write("Ask questions and retrieve answers with precise timestamps from YouTube videos.")
st.write("---")

# Split the main panel layout: Left for Chat, Right for Video Player & DB statistics
col_chat, col_context = st.columns([6, 5])


# ============================================================================
# LEFT COLUMN: INTERACTIVE CHAT WINDOW
# ============================================================================
with col_chat:
    st.subheader("💬 Chat Thread")
    
    # 1. Chat rendering box
    chat_container = st.container(height=500, border=True)
    
    with chat_container:
        if st.session_state.messages:
            # Render each message stored in session history
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message user-message">👤 <b>You:</b> {msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message assistant-message">🤖 <b>Assistant:</b><br>{msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("Chat thread is empty. Load a video and submit a question to start.")
            
    st.write("---")
    
    # 2. Query input form (visible only when a video is processed)
    if st.session_state.video_processed:
        user_query = st.text_area(
            "Enter your question about the video:",
            placeholder="e.g., What is a while loop?",
            height=80,
            key=f"chat_text_box_{st.session_state.chat_input_key}",
            help="Ask anything related to the content mentioned in the video."
        )
        
        col_send, col_clear = st.columns([4, 1])
        with col_send:
            submit_btn = st.button("Query RAG Assistant 📤", use_container_width=True)
        with col_clear:
            clear_btn = st.button("Clear Text 🧹", use_container_width=True)
            
        # Action: Increment key to clear text box and refresh Streamlit layout
        if clear_btn:
            st.session_state.chat_input_key += 1
            st.rerun()
            
        # 3. Query execution flow
        if submit_btn and user_query:
            # Clear the ready status message once the user starts chatting
            st.session_state.show_ready_message = False
            
            # Store user message in history
            st.session_state.messages.append({
                "role": "user",
                "content": user_query,
                "timestamp": datetime.now().isoformat()
            })
            
            with st.spinner("Searching database and generating response..."):
                try:
                    # check if the user query is asking for a global concepts summary/syllabus
                    is_global = any(kw in user_query.lower() for kw in [
                        "concept", "syllabus", "topics", "summary", "summarize", 
                        "overview", "outline", "chapters", "agenda", "content covered", 
                        "all the things taught", "what is this video about", "what does this video cover"
                    ])
                    
                    if is_global:
                        # Global RAG: Analyze entire timed transcript directly in Gemini
                        assistant_response = extract_video_concepts(
                            st.session_state.transcript, 
                            query=user_query, 
                            model_name=model_name
                        )
                        retrieved_chunks = []
                    else:
                        # Standard RAG: Retrieve top matching chunks from local Vector DB
                        retrieved_chunks = search_query(user_query, st.session_state.db)
                        
                        if not retrieved_chunks:
                            assistant_response = "Not mentioned in video"
                        else:
                            # Refine chunk start timestamp to exact sentence coordinate
                            best_chunk_start = retrieved_chunks[0]["start"]
                            precise_time = choose_precise_timestamp(
                                user_query, st.session_state.transcript, best_chunk_start
                            )
                            
                            # Combine retrieved chunks to build context window
                            text_context = "\n".join([c["text"] for c in retrieved_chunks[:3]])
                            
                            # Query Gemini with strict RAG prompt
                            assistant_response = answer_with_context(
                                question=user_query,
                                context=text_context,
                                timestamp=precise_time,
                                model_name=model_name
                            )
                            
                    # Store assistant message in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "timestamp": datetime.now().isoformat(),
                        "retrieved": retrieved_chunks
                    })
                    
                    # Refresh widget and trigger reload
                    st.session_state.chat_input_key += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"RAG system error: {str(e)}")
    else:
        st.warning("⚠️ RAG database not initialized. Paste a YouTube URL and click 'Process Video' first.")


# ============================================================================
# RIGHT COLUMN: VIDEO PLAYER & RETRIEVAL TRACE
# ============================================================================
with col_context:
    st.subheader("📺 Video Player & Source Data")
    
    # 1. Video Player: Embeds the native YouTube player in the column
    if st.session_state.youtube_url:
        st.video(st.session_state.youtube_url)
        st.info("💡 Tip: You can jump to timestamps returned in the chat box inside the video player above.")
    else:
        st.info("No active video. Load a YouTube link from the sidebar.")

    st.write("---")
    st.subheader("🔍 Retrieval Trace")

    # 2. RAG Trace panel: Displays the exact chunks retrieved from database for verification
    if st.session_state.messages:
        # Fetch the last assistant response from history
        latest_assistant = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                latest_assistant = msg
                break
                
        # If the query had retrieved chunks, display them in neat expanders
        if latest_assistant and latest_assistant.get("retrieved"):
            st.success("✅ Top contexts retrieved from all-MiniLM-L6-v2 database:")
            for idx, doc in enumerate(latest_assistant["retrieved"]):
                # Format start second into readable timestamp
                from summarizer import format_timestamp
                formatted_start = format_timestamp(doc["start"])
                
                with st.expander(f"Chunk #{idx+1} (Timestamp: {formatted_start}) - Match Distance: {doc['score']:.4f}"):
                    st.write(doc["text"])
        else:
            st.write("Submit a question to see the matching contexts retrieved by semantic search.")
    else:
        st.write("No queries executed yet.")
