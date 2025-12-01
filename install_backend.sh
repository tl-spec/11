#!/bin/bash
# install_backend.sh
# This script sets up the backend environment from the project root.
# It creates a virtual environment named "insightagent-venv", activates it,
# upgrades essential build tools, and then installs the backend project
# (located in the "server" directory) in editable mode.
#
# Important: Run this script using 'source install_backend.sh' (or '. install_backend.sh')
# so that the virtual environment remains active in your current shell.

# Step 1: Create the virtual environment if it doesn't exist.
if [ ! -d "insightagent-venv" ]; then
    echo "Creating virtual environment 'insightagent-venv'..."
    python -m venv insightagent-venv
else
    echo "Virtual environment 'insightagent-venv' already exists."
fi

# Step 2: Activate the virtual environment.
echo "Activating virtual environment 'insightagent-venv'..."
if [ -f "insightagent-venv/bin/activate" ]; then
    source insightagent-venv/bin/activate
else
    echo "Error: Cannot find insightagent-venv/bin/activate. Exiting."
    exit 1
fi

# Step 3: Upgrade pip, setuptools, and wheel to ensure a smooth installation.
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Step 4: Verify that the 'server' directory exists.
if [ ! -d "server" ]; then
    echo "Error: 'server' directory not found. Please ensure it exists in the project root."
    exit 1
fi

echo "Changing directory to 'server'..."
cd server

# Step 5: Install the backend project in editable mode.
# This should also trigger any custom install commands in the server's setup.py,
# such as installing the ChatFlare submodule.
echo "Installing the backend project in editable mode..."
pip install -e .
pip install -e ChatFlare

deactivate

cd ..
echo "✅ Backend installation complete!"
echo "The virtual environment 'insightagent-venv' is now active."

