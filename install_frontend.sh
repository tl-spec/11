#!/bin/bash
# install_frontend.sh
# This script checks for Node.js and pnpm, installs the front-end dependencies, and launches the dev server.

# Check if Node.js is installed
if ! command -v node >/dev/null 2>&1; then
    echo "❌ Node.js is not installed. Please install Node.js (recommended version: 20.19.0)."
    echo "You can use nvm (https://github.com/nvm-sh/nvm) or Volta (https://volta.sh) to manage Node versions."
    exit 1
fi

# Check Node.js version
REQUIRED_NODE_VERSION="20.19.0"
NODE_VERSION=$(node --version | sed 's/v//')
echo "Detected Node.js version: $NODE_VERSION"

# Simple version comparison using sort (works for basic cases)
if [[ "$(printf '%s\n' "$REQUIRED_NODE_VERSION" "$NODE_VERSION" | sort -V | head -n1)" != "$REQUIRED_NODE_VERSION" ]]; then
    echo "❌ Your Node.js version is lower than the required version ($REQUIRED_NODE_VERSION). Please upgrade Node.js."
    exit 1
fi

# Check if pnpm is installed
if ! command -v pnpm >/dev/null 2>&1; then
    echo "⚠️ pnpm is not found. Installing pnpm globally using npm..."
    npm install -g pnpm
fi

echo "✅ Node.js and pnpm check passed."

# Install front-end dependencies
echo "📦 Installing front-end dependencies..."
pnpm install

# Install for play directory 
cd play
pnpm install
cd ..

# Install firebase tools 
npm install -g firebase-tools

echo "✅ Frontend installation complete!"
echo "You can now start the development server with 'pnpm run dev'"