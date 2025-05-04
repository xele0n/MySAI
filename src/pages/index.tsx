import { useState } from 'react'
import Head from 'next/head'
import Link from 'next/link'

export default function Home() {
  return (
    <>
      <Head>
        <title>MySAI - AI Playground</title>
        <meta name="description" content="AI Playground with chatbot and learning simulation" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="flex min-h-screen flex-col items-center justify-center p-24">
        <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm">
          <h1 className="text-4xl font-bold text-center mb-10">MySAI - AI Playground</h1>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <Link href="/your-ai" className="group">
              <div className="rounded-lg border border-gray-300 bg-white p-8 dark:border-neutral-700 dark:bg-neutral-800/30 transition-all hover:border-blue-500 hover:bg-blue-100 dark:hover:border-blue-500 dark:hover:bg-blue-900/30">
                <h2 className="mb-3 text-2xl font-semibold">
                  Your AI{' '}
                  <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                    -&gt;
                  </span>
                </h2>
                <p className="m-0 max-w-[30ch] text-sm opacity-80">
                  Chat with AI, upload images, and train your AI by consuming data from the internet.
                </p>
              </div>
            </Link>

            <Link href="/ai-playground" className="group">
              <div className="rounded-lg border border-gray-300 bg-white p-8 dark:border-neutral-700 dark:bg-neutral-800/30 transition-all hover:border-blue-500 hover:bg-blue-100 dark:hover:border-blue-500 dark:hover:bg-blue-900/30">
                <h2 className="mb-3 text-2xl font-semibold">
                  AI Playground{' '}
                  <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                    -&gt;
                  </span>
                </h2>
                <p className="m-0 max-w-[30ch] text-sm opacity-80">
                  Machine learning simulation showing AI learning to walk through trial and error.
                </p>
              </div>
            </Link>
          </div>
        </div>
      </main>
    </>
  )
} 