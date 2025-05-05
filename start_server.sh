#!/bin/bash

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Starting the AI Playground Python server..."
python server.py 