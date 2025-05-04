# MySAI - AI Playground Setup Guide

## Installation

### Prerequisites
- Node.js (v14 or newer)
- npm (comes with Node.js)

### Steps

#### Windows
1. Double-click the `scripts/setup.bat` file to install dependencies
2. Run `start.bat` to start the application

#### Mac/Linux
1. Open a terminal in the project directory
2. Run `chmod +x scripts/setup.sh` to make the setup script executable
3. Run `./scripts/setup.sh` to install dependencies
4. Run `npm run dev` to start the application

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

## Project Structure
- `/src/pages/index.tsx` - Main menu page
- `/src/pages/your-ai.tsx` - AI chatbot interface
- `/src/pages/ai-playground.tsx` - AI learning simulation

## Development
- Run `npm run dev` for development
- Run `npm run build` to build the application
- Run `npm start` to start the production server 