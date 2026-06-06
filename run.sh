#!/bin/bash
# ============================================================================
# START BACKEND AND FRONTEND
# ============================================================================
# This script starts both FastAPI backend and Streamlit frontend on Mac/Linux
#
# What it does:
# 1. Activates Python virtual environment
# 2. Installs dependencies if needed
# 3. Starts FastAPI backend on port 8000
# 4. Starts Streamlit frontend on port 8501
#
# Usage:
#   chmod +x run.sh  # Make executable (first time only)
#   ./run.sh         # Run the script
#
# Requirements:
#   - Python 3.8+
#   - Virtual environment (venv folder)

set -e  # Exit on error

echo ""
echo "============================================================"
echo "  YouTube RAG Assistant - Startup Script"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Install Python from: https://www.python.org/downloads/"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies (this may take a few minutes)..."
pip install -q -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f "backend/.env" ]; then
    echo "Creating backend/.env file..."
    cat > backend/.env << 'EOF'
# Add your Google API Key here
GOOGLE_API_KEY=your_api_key_here
LLM_MODEL=gemini-pro
DATABASE_URL=sqlite:///./conversations.db
EOF
    echo ""
    echo "IMPORTANT: Edit backend/.env and add your GOOGLE_API_KEY"
    echo "Get one from: https://makersuite.google.com/app/apikey"
    echo ""
fi

echo ""
echo "============================================================"
echo "  Starting Services..."
echo "============================================================"
echo ""
echo "This will start two processes:"
echo "  1. Backend FastAPI on http://localhost:8000"
echo "  2. Frontend Streamlit on http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup SIGINT

# Start backend in background
echo "Starting Backend (FastAPI)..."
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
sleep 2

# Start frontend in background
echo "Starting Frontend (Streamlit)..."
streamlit run streamlit_app.py > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
sleep 2

echo ""
echo "============================================================"
echo "  Services Started!"
echo "============================================================"
echo ""
echo "Backend API:      http://localhost:8000"
echo "API Docs:         http://localhost:8000/docs"
echo "Frontend:         http://localhost:8501"
echo ""
echo "Logs:"
echo "  Backend:  tail -f /tmp/backend.log"
echo "  Frontend: tail -f /tmp/frontend.log"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for both processes
wait
