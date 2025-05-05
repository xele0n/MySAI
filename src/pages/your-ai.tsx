import { useState, useRef, useEffect, ChangeEvent } from 'react'
import Head from 'next/head'
import Link from 'next/link'

type Message = {
  role: 'user' | 'ai'
  content: string
  timestamp: Date
  image?: string
}

type KaggleDataset = {
  ref: string
  title: string
  subtitle: string
  size: string
}

export default function YourAI() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'ai' as const,
      content: 'Hello! I am your AI assistant. How can I help you today? You can also train me with Kaggle datasets!',
      timestamp: new Date()
    }
  ])
  const [input, setInput] = useState('')
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [isServerConnected, setIsServerConnected] = useState(true)
  const [kaggleDatasets, setKaggleDatasets] = useState<KaggleDataset[]>([])
  const [isDatasetModalOpen, setIsDatasetModalOpen] = useState(false)
  const [selectedDataset, setSelectedDataset] = useState<KaggleDataset | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  // Fetch Kaggle datasets when the component mounts
  useEffect(() => {
    fetchKaggleDatasets();
  }, []);

  // Poll training status if training is in progress
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isTraining) {
      interval = setInterval(fetchTrainingStatus, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isTraining]);

  const fetchKaggleDatasets = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:5001/api/kaggle/datasets');
      
      if (!response.ok) {
        throw new Error(`Failed to fetch datasets: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setKaggleDatasets(data.datasets);
      setIsServerConnected(true);
    } catch (error) {
      console.error('Error fetching Kaggle datasets:', error);
      setError(`Failed to connect to AI server: ${error instanceof Error ? error.message : String(error)}`);
      setIsServerConnected(false);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/train/status');
      
      if (!response.ok) {
        throw new Error(`Failed to fetch training status: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setTrainingProgress(data.progress);
      setIsTraining(data.is_training);
      
      // If training completed, add a message
      if (data.progress === 100 && !data.is_training) {
        setMessages(prev => [
          ...prev,
          {
            role: 'ai' as const,
            content: 'Training complete! I have improved my knowledge based on the dataset.',
            timestamp: new Date()
          }
        ]);
      }
    } catch (error) {
      console.error('Error fetching training status:', error);
      setIsTraining(false);
    }
  };

  const handleSendMessage = async () => {
    if (input.trim() === '') return;

    // Add user message
    const newMessages = [
      ...messages,
      {
        role: 'user' as const,
        content: input,
        timestamp: new Date()
      }
    ];
    setMessages(newMessages);
    const userInput = input;
    setInput('');
    setIsLoading(true);

    try {
      // Call AI chat API
      const response = await fetch('http://localhost:5001/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userInput,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get response: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Add AI response
      setMessages(prev => [
        ...prev,
        {
          role: 'ai' as const,
          content: data.response,
          timestamp: new Date()
        }
      ]);
      setIsServerConnected(true);
    } catch (error) {
      console.error('Error sending message:', error);
      setError(`Failed to get AI response: ${error instanceof Error ? error.message : String(error)}`);
      setIsServerConnected(false);
      
      // Add error message
      setMessages(prev => [
        ...prev,
        {
          role: 'ai' as const,
          content: 'Sorry, I encountered an error. Please check if the AI server is running.',
          timestamp: new Date()
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (event) => {
      const imageUrl = event.target?.result as string;
      
      // Add message with image
      setMessages(prev => [
        ...prev,
        {
          role: 'user' as const,
          content: 'I uploaded an image.',
          image: imageUrl,
          timestamp: new Date()
        }
      ]);

      setIsLoading(true);

      try {
        // Call AI chat API with image
        const response = await fetch('http://localhost:5001/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: 'I uploaded an image.',
            image: imageUrl,
          }),
        });
        
        if (!response.ok) {
          throw new Error(`Failed to process image: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Add AI response
        setMessages(prev => [
          ...prev,
          {
            role: 'ai' as const,
            content: data.response,
            timestamp: new Date()
          }
        ]);
        setIsServerConnected(true);
      } catch (error) {
        console.error('Error processing image:', error);
        setError(`Failed to process image: ${error instanceof Error ? error.message : String(error)}`);
        setIsServerConnected(false);
        
        // Add error message
        setMessages(prev => [
          ...prev,
          {
            role: 'ai' as const,
            content: 'Sorry, I encountered an error processing your image. Please check if the AI server is running.',
            timestamp: new Date()
          }
        ]);
      } finally {
        setIsLoading(false);
      }
    };
    reader.readAsDataURL(file);
  };

  const startTraining = async (dataset: KaggleDataset) => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:5001/api/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dataset_ref: dataset.ref,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to start training: ${response.status} ${response.statusText}`);
      }
      
      setIsTraining(true);
      setTrainingProgress(0);
      setIsDatasetModalOpen(false);
      
      // Add message about training
      setMessages(prev => [
        ...prev,
        {
          role: 'ai' as const,
          content: `I am now training on the ${dataset.title} dataset. This will enhance my knowledge!`,
          timestamp: new Date()
        }
      ]);
      
    } catch (error) {
      console.error('Error starting training:', error);
      setError(`Failed to start training: ${error instanceof Error ? error.message : String(error)}`);
      
      // Add error message
      setMessages(prev => [
        ...prev,
        {
          role: 'ai' as const,
          content: 'Sorry, I encountered an error starting the training process.',
          timestamp: new Date()
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const cancelTraining = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/train/cancel', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to cancel training: ${response.status} ${response.statusText}`);
      }
      
      setIsTraining(false);
      
      // Add message about cancellation
      setMessages(prev => [
        ...prev,
        {
          role: 'ai' as const,
          content: 'Training has been cancelled.',
          timestamp: new Date()
        }
      ]);
      
    } catch (error) {
      console.error('Error cancelling training:', error);
      
      // Keep trying to cancel
      setIsTraining(false);
    }
  };

  return (
    <>
      <Head>
        <title>Your AI - MySAI Playground</title>
        <meta name="description" content="Chat with your AI assistant" />
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
          <h1 className="text-xl font-bold">Your AI</h1>
          <div className="flex gap-2">
            <button
              onClick={() => setIsDatasetModalOpen(true)}
              disabled={isTraining}
              className={`px-4 py-2 rounded ${
                isTraining 
                  ? 'bg-gray-300 cursor-not-allowed' 
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              Train AI
            </button>
            {isTraining && (
              <button
                onClick={cancelTraining}
                className="px-4 py-2 rounded bg-red-500 text-white hover:bg-red-600"
              >
                Cancel Training
              </button>
            )}
          </div>
        </div>

        {isTraining && (
          <div className="p-4 bg-blue-50 dark:bg-blue-900/30">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full"
                style={{ width: `${trainingProgress}%` }}
              ></div>
            </div>
            <p className="text-center mt-2 text-sm">Training progress: {trainingProgress}%</p>
          </div>
        )}

        {error && !isServerConnected && (
          <div className="p-4 bg-red-50 dark:bg-red-900/30 border-b border-red-200 dark:border-red-900">
            <p className="text-center text-red-600 dark:text-red-400">
              <strong>Connection Error:</strong> {error}
            </p>
            <p className="text-center text-sm mt-1">
              Make sure the chat server is running: <code>python chat_server.py</code>
            </p>
          </div>
        )}

        <div className="flex-1 overflow-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  message.role === 'user'
                    ? 'bg-blue-500 text-white dark:bg-blue-600'
                    : 'bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                }`}
              >
                {message.image && (
                  <img
                    src={message.image}
                    alt="User uploaded"
                    className="max-w-full rounded mb-2"
                  />
                )}
                <p>{message.content}</p>
                <p className="text-xs opacity-70 mt-1">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200 rounded-lg px-4 py-2 max-w-[80%]">
                <div className="flex space-x-2">
                  <div className="w-3 h-3 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-3 h-3 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-3 h-3 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="border-t p-4 bg-white dark:bg-gray-800">
          <div className="flex space-x-2">
            <button
              className="p-2 rounded bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept="image/*"
                className="hidden"
                disabled={isLoading}
              />
            </button>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSendMessage()}
              placeholder="Type a message..."
              className="flex-1 p-2 border rounded-l focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              disabled={isLoading}
            />
            <button
              onClick={handleSendMessage}
              className={`bg-blue-500 text-white px-4 py-2 rounded-r hover:bg-blue-600 ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
              disabled={isLoading}
            >
              Send
            </button>
          </div>
        </div>
      </main>

      {/* Kaggle Dataset Selection Modal */}
      {isDatasetModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-lg w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="p-4 border-b flex justify-between items-center">
              <h2 className="text-xl font-bold">Select Kaggle Dataset for Training</h2>
              <button 
                onClick={() => setIsDatasetModalOpen(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="overflow-y-auto flex-1 p-4">
              {kaggleDatasets.length === 0 ? (
                <p className="text-center text-gray-500">Loading datasets...</p>
              ) : (
                <div className="space-y-3">
                  {kaggleDatasets.map((dataset, index) => (
                    <div 
                      key={index}
                      className={`p-3 border rounded cursor-pointer hover:bg-blue-50 dark:hover:bg-blue-900/30 ${
                        selectedDataset?.ref === dataset.ref ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30' : ''
                      }`}
                      onClick={() => setSelectedDataset(dataset)}
                    >
                      <h3 className="font-medium">{dataset.title}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{dataset.subtitle}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">Size: {dataset.size}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            <div className="p-4 border-t flex justify-end space-x-3">
              <button
                onClick={() => setIsDatasetModalOpen(false)}
                className="px-4 py-2 border rounded hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={() => selectedDataset && startTraining(selectedDataset)}
                disabled={!selectedDataset || isLoading}
                className={`px-4 py-2 rounded ${
                  !selectedDataset || isLoading
                    ? 'bg-gray-300 cursor-not-allowed'
                    : 'bg-blue-500 text-white hover:bg-blue-600'
                }`}
              >
                Start Training
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
} 