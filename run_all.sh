#!/bin/bash
# run_all.sh
# This script starts Firebase emulators, the backend server, and the frontend in the background.

# --- Step 1: Start Firebase emulators ---
echo "Starting Firebase emulators..."
nohup firebase emulators:start > firebase_emulators.log 2>&1 &
EMULATOR_PID=$!
echo "Firebase emulators started with PID: $EMULATOR_PID (logs: firebase_emulators.log)"

# --- Step 2: Start the backend ---
echo "Starting backend server..."
if [ ! -d "server" ]; then
    echo "Error: 'server' directory not found!"
    exit 1
fi

cd server || exit 1

# Activate the virtual environment 'insightagent-venv' (assuming it is located in the project root)
if [ -f "../insightagent-venv/bin/activate" ]; then
    echo "Activating virtual environment 'insightagent-venv'..."
    source ../insightagent-venv/bin/activate
else
    echo "Error: Virtual environment '../insightagent-venv' not found. Please create it first."
    exit 1
fi

# Run the backend server (app.py) in the background and redirect output to a log file.
echo "Launching backend (app.py)..."
nohup python app.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID (logs in server/backend.log)"

# Deactivate the virtual environment and return to the root directory.
deactivate
cd ..

# --- Step 3: Start the frontend ---
echo "Starting frontend..."
nohup npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID (logs in frontend.log)"

echo "All services have been started in the background."
echo "Check logs: Firebase (firebase_emulators.log), Backend (server/backend.log), Frontend (frontend.log)."
