#!/bin/bash
# install_all.sh
# Master installation script to set up the entire project.
# Order of execution:
#   1. Install backend (creates & activates "insideagent-venv", installs backend + ChatFlare)
#   2. Install frontend (checks Node.js/pnpm, installs front-end dependencies, etc.)
#   3. Install functions (creates virtual environment inside functions and installs dependencies)
#
# IMPORTANT: Run this script via 'source install_all.sh' or '. install_all.sh'
# so that any environment changes persist in your current shell if needed.

# Check existence of required scripts
if [ ! -f "install_backend.sh" ]; then
    echo "Error: install_backend.sh not found in the project root!"
    exit 1
fi
if [ ! -f "install_frontend.sh" ]; then
    echo "Error: install_frontend.sh not found in the project root!"
    exit 1
fi
if [ ! -f "install_functions.sh" ]; then
    echo "Error: install_functions.sh not found in the project root!"
    exit 1
fi

echo "======================================"
echo "Installing BACKEND..."
echo "======================================"
# Use source to run the install script so that the venv activation remains active if needed.
source install_backend.sh
echo "Backend installation complete!"
echo ""

echo "======================================"
echo "Installing FRONTEND..."
echo "======================================"
source install_frontend.sh
echo "Frontend installation complete!"
echo ""

echo "======================================"
echo "Installing FUNCTIONS environment..."
echo "======================================"
source install_functions.sh
echo "Functions environment installation complete!"
echo ""

echo "All installations complete!"
