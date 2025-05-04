import { useState, useRef, ChangeEvent } from 'react'
import Head from 'next/head'
import Link from 'next/link'

type Message = {
  role: 'user' | 'ai'
  content: string
  timestamp: Date
  image?: string
}

export default function YourAI() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'ai',
      content: 'Hello! I am your AI assistant. How can I help you today?',
      timestamp: new Date()
    }
  ])
  const [input, setInput] = useState('')
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSendMessage = () => {
    if (input.trim() === '') return

    // Add user message
    const newMessages = [
      ...messages,
      {
        role: 'user' as const,
        content: input,
        timestamp: new Date()
      }
    ]
    setMessages(newMessages)
    setInput('')

    // Simulate AI response (in a real app, this would call an API)
    setTimeout(() => {
      setMessages(prev => [
        ...prev,
        {
          role: 'ai' as const,
          content: `I received your message: "${input}". This is a simulated response as this is a demo.`,
          timestamp: new Date()
        }
      ])
    }, 1000)
  }

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      const imageUrl = event.target?.result as string
      
      // Add message with image
      setMessages(prev => [
        ...prev,
        {
          role: 'user' as const,
          content: 'I uploaded an image.',
          image: imageUrl,
          timestamp: new Date()
        }
      ])

      // Simulate AI response to image
      setTimeout(() => {
        setMessages(prev => [
          ...prev,
          {
            role: 'ai' as const,
            content: 'I received your image. This is a simulated image analysis response.',
            timestamp: new Date()
          }
        ])
      }, 1500)
    }
    reader.readAsDataURL(file)
  }

  const startTraining = () => {
    setIsTraining(true)
    setTrainingProgress(0)

    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsTraining(false)
          
          // Add training completion message
          setMessages(prev => [
            ...prev,
            {
              role: 'ai' as const,
              content: 'Training complete! I have improved my knowledge based on the latest data.',
              timestamp: new Date()
            }
          ])
          
          return 0
        }
        return prev + 5
      })
    }, 300)
  }

  return (
    <>
      <Head>
        <title>Your AI - MySAI Playground</title>
        <meta name="description" content="Chat with your AI assistant" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="flex min-h-screen flex-col">
        <div className="flex items-center justify-between p-4 border-b">
          <Link href="/" className="text-blue-500 hover:underline flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
            </svg>
            Back to Home
          </Link>
          <h1 className="text-xl font-bold">Your AI</h1>
          <button
            onClick={startTraining}
            disabled={isTraining}
            className={`px-4 py-2 rounded ${
              isTraining 
                ? 'bg-gray-300 cursor-not-allowed' 
                : 'bg-blue-500 text-white hover:bg-blue-600'
            }`}
          >
            {isTraining ? 'Training...' : 'Train AI'}
          </button>
        </div>

        {isTraining && (
          <div className="p-4 bg-blue-50">
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full"
                style={{ width: `${trainingProgress}%` }}
              ></div>
            </div>
            <p className="text-center mt-2 text-sm">Training progress: {trainingProgress}%</p>
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
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-800'
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
        </div>

        <div className="border-t p-4">
          <div className="flex space-x-2">
            <button
              className="p-2 rounded bg-gray-200 hover:bg-gray-300"
              onClick={() => fileInputRef.current?.click()}
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
              />
            </button>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type a message..."
              className="flex-1 p-2 border rounded-l focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={handleSendMessage}
              className="bg-blue-500 text-white px-4 py-2 rounded-r hover:bg-blue-600"
            >
              Send
            </button>
          </div>
        </div>
      </main>
    </>
  )
} 