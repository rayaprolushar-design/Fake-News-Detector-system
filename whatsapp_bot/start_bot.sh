#!/bin/bash

# Ensure background server process is killed when this script exits
trap "echo -e '\nStopping FastAPI server...'; kill 0" EXIT

echo "========================================="
echo "   🚀 Starting VerifyAI WhatsApp Bot"
echo "========================================="

# 1. Start the FastAPI server in the background
echo "Starting server process..."
cd "$(dirname "$0")"
../.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
echo "FastAPI Server started (PID: $SERVER_PID)"

# 2. Expose the port and prevent sleep on macOS
echo "Opening public tunnel. Keeping Mac awake using 'caffeinate'..."
echo "Press Ctrl+C to stop the bot and shutdown the server."
echo "-----------------------------------------"

caffeinate -i npx -y localtunnel --port 8000
