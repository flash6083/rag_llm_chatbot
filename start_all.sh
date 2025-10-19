#!/bin/bash

echo "🚀 Starting NIT Rourkela CS Chatbot..."

# Start ChromaDB embedded server in background
echo "🟢 Starting ChromaDB server..."
npx chroma run --path ./chroma_db &
CHROMA_PID=$!
echo "✅ ChromaDB server started with PID $CHROMA_PID"

# Wait a few seconds for Chroma to initialize
sleep 5

# Start Python service in background
echo "🐍 Starting Python microservice..."
cd python_service
source venv/bin/activate
python app.py &
PYTHON_PID=$!
cd ..

# Wait for Python service to start
sleep 5

# Start Node.js backend
echo "📡 Starting Node.js backend..."
npm run dev &
NODE_PID=$!

# Cleanup on exit
trap "echo '🛑 Shutting down...'; kill $CHROMA_PID $PYTHON_PID $NODE_PID" EXIT

# Wait for Node.js to exit
wait $NODE_PID
