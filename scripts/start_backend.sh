#!/bin/bash
# Start FastAPI Backend Server
echo "Starting FPL AI Optimizer Backend..."
echo ""

cd "$(dirname "$0")/.."
python -m uvicorn backend.src.api.main:app --reload --host 0.0.0.0 --port 8000
