import os
import sys
import json
import base64
import random
import time
import traceback
import signal
import threading
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

print("Starting Enhanced AI Chatbot Server")

# Global variables
topic_knowledge = {}
training_data = {}
training_thread = None
is_training = False
training_progress = 0
kaggle_datasets = []
current_topic = ""

# For demo version, we'll use predefined responses instead of actual ML models
demo_responses = [
    "That's an interesting question! Let me think about that.",
    "I understand what you're asking. Here's what I know about that topic.",
    "Thanks for sharing that with me. I'd like to learn more.",
    "Based on my understanding, I think the answer is multi-faceted.",
    "That's a great point you've raised. I've been considering similar ideas.",
    "I'm designed to help with questions like that. Here's my perspective.",
    "I appreciate your curiosity! That's an area worth exploring further.",
    "Let me process that request for you. I have some thoughts on that.",
    "I find your question fascinating. Let's explore it together.",
    "That's an area where I can provide some insights based on my training."
]

# Topic-specific responses
topic_responses = {}

def chat_response(message, image_data=None):
    try:
        # Handle image if provided
        image_description = ""
        if image_data:
            try:
                # Just acknowledge the image was received
                image_description = "I can see an image you've uploaded. "
                image_description += random.choice([
                    "It appears to be a photograph.",
                    "It looks like a digital image.",
                    "I can process visual information from this image.",
                    "Thank you for sharing this visual content."
                ])
            except Exception as e:
                print(f"Error processing image: {e}")
                image_description = "I had trouble processing your image."
        
        # Check if the response should be topic-specific
        global current_topic
        if current_topic and current_topic in topic_responses and topic_responses[current_topic]:
            # Use a topic-specific response
            response = random.choice(topic_responses[current_topic])
            response = f"[{current_topic}] {response}"
        else:
            # Generate a standard demo response
            response = random.choice(demo_responses)
        
        # Add personalization based on the message
        if "?" in message:
            response = "That's a good question! " + response
        if "help" in message.lower():
            response = "I'm here to help. " + response
        if "thanks" in message.lower() or "thank you" in message.lower():
            response = "You're welcome! " + response
        
        # Add image description if there was an image
        if image_description:
            response = image_description + " " + response
            
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        traceback.print_exc()
        return "I'm having trouble processing your request right now."

def generate_image(prompt):
    try:
        # Create a matplotlib figure for the generated image
        plt.figure(figsize=(8, 6))
        
        # Generate a random abstract image based on the prompt
        # Convert prompt to a seed for reproducibility
        seed = sum(ord(c) for c in prompt)
        random.seed(seed)
        np.random.seed(seed)
        
        # Create a colorful abstract image
        if "landscape" in prompt.lower():
            # Generate a landscape-like image
            x = np.linspace(0, 10, 1000)
            y = np.sin(x) + np.random.normal(0, 0.1, len(x))
            plt.plot(x, y, 'b-')
            plt.fill_between(x, y, alpha=0.3)
            plt.title(f"Generated Landscape: {prompt}")
        
        elif "portrait" in prompt.lower():
            # Generate a portrait-like image
            data = np.random.normal(0, 1, (20, 20))
            plt.imshow(data, cmap="viridis")
            plt.title(f"Generated Portrait: {prompt}")
            
        else:
            # Generate an abstract art piece
            data = np.random.rand(10, 10)
            for i in range(5):
                data = np.repeat(np.repeat(data, 2, axis=0), 2, axis=1)
                data += np.random.rand(*data.shape) * 0.1
            
            plt.imshow(data, cmap="plasma")
            plt.title(f"Generated Art: {prompt}")
        
        plt.axis('off')
        
        # Save the image to a BytesIO object
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        plt.close()
        
        return img_bytes
        
    except Exception as e:
        print(f"Error generating image: {e}")
        traceback.print_exc()
        return None

def fetch_kaggle_datasets():
    global kaggle_datasets
    
    try:
        # For demo, we'll use static datasets that don't require Kaggle API
        kaggle_datasets = [
            {
                "ref": "mnist_784",
                "title": "MNIST Handwritten Digits",
                "subtitle": "The famous handwritten digits dataset",
                "size": "11MB"
            },
            {
                "ref": "iris",
                "title": "Iris Flower Dataset",
                "subtitle": "Famous classification dataset",
                "size": "4KB"
            },
            {
                "ref": "titanic",
                "title": "Titanic: Machine Learning from Disaster",
                "subtitle": "Predict survival on the Titanic",
                "size": "61KB"
            },
            {
                "ref": "custom_topic",
                "title": "Custom Topic Learning",
                "subtitle": "Train the AI on any topic you choose",
                "size": "N/A"
            }
        ]
        return kaggle_datasets
            
    except Exception as e:
        print(f"Error fetching Kaggle datasets: {e}")
        traceback.print_exc()
        kaggle_datasets = []
        return kaggle_datasets

def generate_topic_facts(topic):
    """Generate facts about a specific topic using pandas for data handling"""
    try:
        # Create a pandas DataFrame to store topic facts
        facts = pd.DataFrame({
            'topic': [topic] * 10,
            'fact_id': range(1, 11),
            'fact': [
                f"Fact 1 about {topic}",
                f"Fact 2 about {topic}",
                f"Fact 3 about {topic}",
                f"Fact 4 about {topic}",
                f"Fact 5 about {topic}",
                f"Interesting discovery about {topic}",
                f"Historical perspective on {topic}",
                f"Future trends for {topic}",
                f"Common misconception about {topic}",
                f"Expert insight about {topic}"
            ],
            'confidence': np.random.uniform(0.7, 0.99, 10)
        })
        
        # Generate some topic-specific responses based on the facts
        responses = [
            f"I've learned that {fact}" for fact in facts['fact']
        ]
        responses.extend([
            f"According to my training on {topic}, there are several interesting aspects to consider.",
            f"I've analyzed information about {topic} and have some insights to share.",
            f"My learning about {topic} suggests that it's a fascinating subject.",
            f"Based on my training data for {topic}, I can offer some perspective."
        ])
        
        return facts, responses
        
    except Exception as e:
        print(f"Error generating topic facts: {e}")
        traceback.print_exc()
        return pd.DataFrame(), []

def train_on_dataset(dataset_ref, custom_topic=None):
    global training_progress, is_training, topic_responses, current_topic
    
    try:
        # Simulate training process
        is_training = True
        training_progress = 0
        
        # Check if this is custom topic training
        if dataset_ref == "custom_topic" and custom_topic:
            print(f"Training on custom topic: {custom_topic}")
            current_topic = custom_topic
            
            # Generate topic-specific facts and responses
            facts, responses = generate_topic_facts(custom_topic)
            
            # Store the responses for later use
            if responses:
                topic_responses[custom_topic] = responses
        
        # Simulate the training process
        for i in range(21):
            if not is_training:  # Allow cancellation
                break
                
            training_progress = i * 5
            print(f"Training progress: {training_progress}%")
            time.sleep(0.5)  # Simulate training time
            
        # Complete training
        if is_training:
            training_progress = 100
            time.sleep(1)
            is_training = False
            print("Training completed")
            
            # If it's a custom topic, save a visualization of the "learning"
            if dataset_ref == "custom_topic" and custom_topic:
                try:
                    # Create a simple visualization of the learning process
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(1, 11), np.random.uniform(0.5, 1.0, 10))
                    plt.title(f"Learning Progress: {custom_topic}")
                    plt.xlabel("Concept")
                    plt.ylabel("Understanding Level")
                    plt.tight_layout()
                    
                    # Save to a file for later reference
                    plt.savefig(f"topic_{custom_topic.replace(' ', '_')}.png")
                    plt.close()
                except Exception as e:
                    print(f"Error creating visualization: {e}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        is_training = False
        training_progress = 0

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        image_data = data.get('image', None)
        
        response = chat_response(message, image_data)
        
        return jsonify({
            'status': 'success',
            'response': response
        })
        
    except Exception as e:
        print(f"Error processing chat request: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/image/generate', methods=['POST'])
def create_image():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({
                'status': 'error',
                'message': 'No prompt provided'
            }), 400
            
        # Generate the image
        img_bytes = generate_image(prompt)
        
        if img_bytes:
            # Convert to base64 for easy transmission
            img_str = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            
            return jsonify({
                'status': 'success',
                'image': f"data:image/png;base64,{img_str}"
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to generate image'
            }), 500
            
    except Exception as e:
        print(f"Error generating image: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/kaggle/datasets', methods=['GET'])
def get_kaggle_datasets():
    try:
        datasets = fetch_kaggle_datasets()
        return jsonify({
            'status': 'success',
            'datasets': datasets
        })
        
    except Exception as e:
        print(f"Error fetching datasets: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/train', methods=['POST'])
def train():
    global training_thread, is_training
    
    try:
        data = request.json
        dataset_ref = data.get('dataset_ref', '')
        custom_topic = data.get('custom_topic', None)
        
        # Check if already training
        if is_training:
            return jsonify({
                'status': 'error',
                'message': 'Training already in progress'
            }), 400
            
        # Start training in background thread
        is_training = True
        training_thread = threading.Thread(
            target=train_on_dataset, 
            args=(dataset_ref, custom_topic)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Training started',
            'topic': custom_topic if custom_topic else dataset_ref
        })
        
    except Exception as e:
        print(f"Error starting training: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/train/status', methods=['GET'])
def train_status():
    try:
        return jsonify({
            'status': 'success',
            'is_training': is_training,
            'progress': training_progress,
            'current_topic': current_topic
        })
        
    except Exception as e:
        print(f"Error getting training status: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/train/cancel', methods=['POST'])
def cancel_training():
    global is_training
    
    try:
        is_training = False
        return jsonify({
            'status': 'success',
            'message': 'Training cancelled'
        })
        
    except Exception as e:
        print(f"Error cancelling training: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Enhanced Chat server is running',
        'endpoints': [
            '/api/chat', 
            '/api/kaggle/datasets', 
            '/api/train',
            '/api/image/generate'
        ]
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'version': '1.1.0'
    })

# Handle graceful shutdown
def signal_handler(sig, frame):
    print('Shutting down server...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    # Fetch Kaggle datasets in background
    threading.Thread(target=fetch_kaggle_datasets).start()
    
    print("Starting Flask server on http://localhost:5001")
    app.run(debug=True, port=5001, threaded=True) 