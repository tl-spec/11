#!/bin/bash
python -m venv insightagent-venv
source insightagent-venv/bin/activate

# install main project
pip install -e .

# install ChatFlare submodule
pip install -e ChatFlare


echo "Starting the server..."
python app.py