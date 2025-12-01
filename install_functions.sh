#!/bin/bash
# install_functions.sh
# This script sets up the Python virtual environment for your Firebase Functions.
# It creates a 'venv' folder under the functions directory and installs its requirements.

# Check if python3.12 is available
if ! command -v python3.12 >/dev/null 2>&1; then
    echo "❌ python3.12 is not installed. Please install Python 3.12 and ensure it is in your PATH."
    exit 1
fi

# Check if the "functions" directory exists in the root
if [ ! -d "functions" ]; then
    echo "❌ Directory 'functions' not found. Please ensure it exists in the root folder."
    exit 1
fi

# Change directory into functions
cd functions

# Create a virtual environment named 'venv' if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment in functions/venv using python3.12..."
    python3.12 -m venv venv
else
    echo "✅ Virtual environment already exists in functions/venv."
fi

# Activate the virtual environment
echo "🚀 Activating the virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Activation script not found at venv/bin/activate. Exiting."
    exit 1
fi

# Install dependencies from requirements11.txt if the file exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies for Firebase Functions..."
    pip install -r requirements.txt
else
    echo "⚠️ No requirements.txt file found in the functions directory."
fi

deactivate
cd ..
echo "🎉 Firebase Functions environment setup is complete!"
