# MySAI - AI Playground

An interactive AI playground application with two main features:
- **Your AI**: Chat with an AI assistant, train it with data from Kaggle, and process images
- **AI Learning Simulation**: Watch AI agents learn to walk through reinforcement learning

## Setup

### Prerequisites
- Node.js (v14 or newer)
- npm (comes with Node.js)
- Python (3.8 or newer)
- pip (comes with Python)

### Installation

1. Clone the repository
2. Install dependencies with:
   ```
   # Install both frontend and backend dependencies
   npm install
   pip install -r requirements.txt
   ```

### Running the application

#### Quick start (Windows)
```
start_servers.bat
```

#### Quick start (Mac/Linux)
```
chmod +x start_servers.sh
./start_servers.sh
```

#### Manual start
1. Start the walking simulation server:
   ```
   python server.py
   ```
2. Start the AI chat server:
   ```
   python chat_server.py
   ```
3. Start the frontend:
   ```
   npm run dev
   ```

### Troubleshooting

If you encounter connection issues or 404 errors:

1. Run the troubleshooting script to check server status:
   ```
   # Windows
   check_servers.bat
   
   # Mac/Linux
   chmod +x check_servers.sh
   ./check_servers.sh
   ```

2. Ensure you're using the correct URLs:
   - Main interface: http://localhost:3000
   - Simulation API: http://localhost:5000
   - Chat API: http://localhost:5001

3. Common issues:
   - **404 errors**: Make sure all three servers are running
   - **Connection refused**: Check if the port is already in use
   - **Blank page**: Wait a few seconds for the Next.js server to compile

4. Manual checking:
   ```
   # Check if servers are responding
   curl http://localhost:5000/health
   curl http://localhost:5001/health
   ```

## Features

### Your AI
- Chat interface with an AI assistant powered by a real language model (TinyLlama)
- Image upload and processing 
- Train the AI with datasets from Kaggle
- Browse and select from a list of popular Kaggle datasets

### AI Playground
- Physics-based simulation of AI learning to walk
- Visual representation of neural networks
- Real-time learning through trial and error
- Powered by NumPy for neural networks and Pymunk for physics simulation

## Project Structure
- `/src/pages/index.tsx` - Main menu page
- `/src/pages/your-ai.tsx` - AI chatbot interface
- `/src/pages/ai-playground.tsx` - AI learning simulation frontend
- `/server.py` - Python backend for the AI walking simulation
- `/chat_server.py` - Python backend for the AI chatbot with Kaggle integration
- `/requirements.txt` - Python dependencies

## Technical Details

The project uses two separate Python backends:

1. **Walking Simulation Backend (server.py)**
   - Uses NumPy for the neural network implementation
   - Uses Pymunk for physics simulation
   - Simulates a genetic algorithm for teaching AI agents to walk
   - Runs on port 5000

2. **AI Chat Backend (chat_server.py)**
   - Uses Hugging Face Transformers for language model
   - Integrates with Kaggle API for dataset access
   - Provides image processing capabilities
   - Runs on port 5001

The frontend is built with:
- Next.js with TypeScript
- Tailwind CSS for styling
- React hooks for state management 