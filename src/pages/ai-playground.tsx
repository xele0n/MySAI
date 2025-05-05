import { useEffect, useRef, useState } from 'react'
import Head from 'next/head'
import Link from 'next/link'

// Define types for our Python API responses
type WalkerBody = {
  x: number
  y: number
  angle: number
}

type Walker = {
  id: number
  bodies: WalkerBody[]
  fitness: number
}

type SimulationState = {
  status: string
  new_generation?: boolean
  generation: number
  best_fitness: number
  walkers: Walker[]
}

export default function AIPlayground() {
  const [generation, setGeneration] = useState(1)
  const [score, setScore] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const requestRef = useRef<number>()
  const [isSimulationRunning, setIsSimulationRunning] = useState(false)
  const [simulationState, setSimulationState] = useState<SimulationState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isServerConnected, setIsServerConnected] = useState(false)
  
  // Initialize the simulation when the component mounts
  useEffect(() => {
    initializeSimulation();
    
    // Set up canvas
    const canvas = canvasRef.current;
    if (!canvas) {
      setError("Canvas element not found");
      return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      setError("Could not get canvas context");
      return;
    }
    
    // Draw initial state with message
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '16px Arial';
    ctx.fillStyle = '#000';
    ctx.textAlign = 'center';
    
    if (!isServerConnected) {
      ctx.fillText('Connecting to simulation server...', canvas.width / 2, canvas.height / 2);
      ctx.fillText('Make sure the Python server is running (start_server.bat/sh)', canvas.width / 2, canvas.height / 2 + 30);
    }
    
    // Set up animation loop
    const animate = () => {
      if (isSimulationRunning) {
        stepSimulation();
        drawSimulation(ctx);
      }
      requestRef.current = requestAnimationFrame(animate);
    };
    
    requestRef.current = requestAnimationFrame(animate);
    
    // Clean up on unmount
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [isSimulationRunning, isServerConnected]);
  
  // Initialize simulation by calling the Python API
  const initializeSimulation = async () => {
    try {
      setError(null);
      const response = await fetch('http://localhost:5000/api/init', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to initialize simulation: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setGeneration(data.generation);
      setScore(data.best_fitness);
      setIsSimulationRunning(true);
      setIsServerConnected(true);
      console.log("Simulation initialized:", data);
    } catch (error) {
      console.error('Error initializing simulation:', error);
      setError(`Failed to connect to simulation server: ${error instanceof Error ? error.message : String(error)}`);
      setIsServerConnected(false);
      
      // Retry connection after 3 seconds
      setTimeout(() => {
        initializeSimulation();
      }, 3000);
    }
  };
  
  // Step the simulation forward by calling the Python API
  const stepSimulation = async () => {
    if (!isServerConnected) return;
    
    try {
      const response = await fetch('http://localhost:5000/api/step');
      
      if (!response.ok) {
        throw new Error(`Failed to step simulation: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json() as SimulationState;
      setSimulationState(data);
      setGeneration(data.generation);
      setScore(data.best_fitness);
    } catch (error) {
      console.error('Error stepping simulation:', error);
      setError(`Simulation step failed: ${error instanceof Error ? error.message : String(error)}`);
      setIsSimulationRunning(false);
      setIsServerConnected(false);
      
      // Retry connection after 3 seconds
      setTimeout(() => {
        initializeSimulation();
      }, 3000);
    }
  };
  
  // Reset the simulation by calling the Python API
  const resetSimulation = async () => {
    try {
      setError(null);
      const response = await fetch('http://localhost:5000/api/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to reset simulation: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setGeneration(data.generation);
      setScore(data.best_fitness);
      setIsServerConnected(true);
      setIsSimulationRunning(true);
      console.log("Simulation reset:", data);
    } catch (error) {
      console.error('Error resetting simulation:', error);
      setError(`Reset failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };
  
  // Draw the current simulation state on the canvas
  const drawSimulation = (ctx: CanvasRenderingContext2D) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // If there's an error, display it and return
    if (error) {
      ctx.font = '16px Arial';
      ctx.fillStyle = 'red';
      ctx.textAlign = 'center';
      
      // Split error message into multiple lines if needed
      const errorLines = error.split('. ');
      let yPos = canvas.height / 2 - (errorLines.length - 1) * 20;
      
      errorLines.forEach(line => {
        ctx.fillText(line, canvas.width / 2, yPos);
        yPos += 40;
      });
      
      return;
    }
    
    // If no simulation state yet, show loading
    if (!simulationState || !simulationState.walkers || simulationState.walkers.length === 0) {
      ctx.font = '16px Arial';
      ctx.fillStyle = '#000';
      ctx.textAlign = 'center';
      ctx.fillText('Simulation running...', canvas.width / 2, canvas.height / 2);
      
      // Draw ground anyway
      ctx.beginPath();
      ctx.moveTo(0, 580);
      ctx.lineTo(800, 580);
      ctx.lineWidth = 5;
      ctx.strokeStyle = '#060a19';
      ctx.stroke();
      
      return;
    }
    
    const { walkers } = simulationState;
    
    // Draw ground
    ctx.beginPath();
    ctx.moveTo(0, 580);
    ctx.lineTo(800, 580);
    ctx.lineWidth = 5;
    ctx.strokeStyle = '#060a19';
    ctx.stroke();
    
    // Draw each walker
    walkers.forEach(walker => {
      if (!walker.bodies || walker.bodies.length < 4) {
        console.warn("Walker missing body parts:", walker);
        return;
      }
      
      const [torso, head, leftLeg, rightLeg] = walker.bodies;
      
      // Draw torso (rectangle)
      ctx.save();
      ctx.translate(torso.x, torso.y);
      ctx.rotate(torso.angle);
      ctx.fillStyle = '#4a5568';
      ctx.fillRect(-25, -40, 50, 80);
      ctx.restore();
      
      // Draw head (circle)
      ctx.beginPath();
      ctx.arc(head.x, head.y, 20, 0, Math.PI * 2);
      ctx.fillStyle = '#4a5568';
      ctx.fill();
      
      // Draw legs (rectangles)
      // Left leg
      ctx.save();
      ctx.translate(leftLeg.x, leftLeg.y);
      ctx.rotate(leftLeg.angle);
      ctx.fillStyle = '#4a5568';
      ctx.fillRect(-7.5, -30, 15, 60);
      ctx.restore();
      
      // Right leg
      ctx.save();
      ctx.translate(rightLeg.x, rightLeg.y);
      ctx.rotate(rightLeg.angle);
      ctx.fillStyle = '#4a5568';
      ctx.fillRect(-7.5, -30, 15, 60);
      ctx.restore();
      
      // Draw joints as small circles
      ctx.beginPath();
      ctx.arc(torso.x, torso.y - 35, 3, 0, Math.PI * 2); // Neck joint
      ctx.arc(torso.x - 20, torso.y + 35, 3, 0, Math.PI * 2); // Left hip
      ctx.arc(torso.x + 20, torso.y + 35, 3, 0, Math.PI * 2); // Right hip
      ctx.fillStyle = '#e53e3e';
      ctx.fill();
    });
  };
  
  return (
    <>
      <Head>
        <title>AI Playground - MySAI</title>
        <meta name="description" content="Watch AI learn to walk through reinforcement learning" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="flex min-h-screen flex-col bg-gray-50 dark:bg-gray-900">
        <div className="flex items-center justify-between p-4 border-b bg-white dark:bg-gray-800 shadow-sm">
          <Link href="/" className="text-blue-500 hover:underline flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
            </svg>
            Back to Home
          </Link>
          <h1 className="text-xl font-bold">AI Learning Simulation</h1>
          <div className="flex gap-2">
            <button
              onClick={resetSimulation}
              className="px-4 py-2 rounded bg-blue-500 text-white hover:bg-blue-600"
            >
              Reset Simulation
            </button>
            <button
              onClick={initializeSimulation}
              className="px-4 py-2 rounded bg-green-500 text-white hover:bg-green-600"
            >
              Reconnect
            </button>
          </div>
        </div>
        
        <div className="flex-1 p-4">
          <div className="flex justify-between mb-4 bg-white dark:bg-gray-800 p-3 rounded shadow-sm">
            <div className="text-lg">
              <span className="font-bold">Generation:</span> {generation}
            </div>
            <div className="text-lg">
              <span className="font-bold">Best Score:</span> {score.toFixed(2)}
            </div>
          </div>
          
          <div className="w-full max-w-4xl mx-auto border border-gray-300 rounded-lg overflow-hidden bg-white dark:bg-gray-800 shadow-md">
            <canvas 
              ref={canvasRef} 
              width={800} 
              height={600}
              className="bg-gray-100 dark:bg-gray-700"
            />
          </div>
          
          <div className="mt-4 max-w-4xl mx-auto text-sm bg-white dark:bg-gray-800 p-4 rounded shadow-sm">
            <p>
              This simulation shows AI agents learning to walk through a genetic algorithm and neural networks.
              Each agent has a neural network that controls its movement. The best performing agents are selected
              to pass their "genes" (neural network weights) to the next generation with slight mutations.
            </p>
            <p className="mt-2">
              <strong>Note:</strong> This simulation runs using a Python backend with numpy for the neural network
              and pymunk for the physics simulation. Make sure the Python server is running at localhost:5000.
            </p>
            {error && (
              <p className="mt-2 text-red-500 font-bold">
                Error: {error}
              </p>
            )}
          </div>
        </div>
      </main>
    </>
  )
} 