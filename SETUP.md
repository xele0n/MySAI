# MySAI - AI Playground Setup Guide

## Installation

### Prerequisites
- Node.js (v14 or newer)
- npm (comes with Node.js)
- Python (3.8 or newer)
- pip (comes with Python)

### Steps

#### Windows
1. Double-click the `scripts/setup.bat` file to install dependencies
2. Run `start_server.bat` to start the Python backend server
3. In a new terminal, run `start.bat` to start the frontend application

#### Mac/Linux
1. Open a terminal in the project directory
2. Run `chmod +x scripts/setup.sh` to make the setup script executable
3. Run `./scripts/setup.sh` to install dependencies
4. Run `chmod +x start_server.sh` to make the server script executable
5. Run `./start_server.sh` to start the Python backend server
6. In a new terminal, run `npm run dev` to start the frontend application

## Python Backend
The AI Playground simulation uses a Python backend with:
- NumPy for neural network implementation
- Flask for the REST API
- Pymunk for physics simulation

All Python dependencies will be installed automatically when you run the start_server script.

## Features

### Your AI
This feature provides a chatbox interface where you can:
- Chat with an AI assistant
- Upload and process images
- Train the AI by consuming data from the internet (simulated)

### AI Playground
This feature provides a physics-based simulation where you can:
- Watch AI agents learn to walk through reinforcement learning
- See the neural network's effect on movement
- Observe how genetic algorithms improve performance over generations
- Real physics simulation using NumPy and Pymunk in a Python backend

## Project Structure
- `/src/pages/index.tsx` - Main menu page
- `/src/pages/your-ai.tsx` - AI chatbot interface
- `/src/pages/ai-playground.tsx` - AI learning simulation frontend
- `/server.py` - Python backend for the AI walking simulation
- `/requirements.txt` - Python dependencies

## Development
- Run `npm run dev` for frontend development
- Run `python server.py` for backend development
- Run `npm run build` to build the application
- Run `npm start` to start the production server 