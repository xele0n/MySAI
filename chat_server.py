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
conversation_history = []  # Store conversation history for context awareness
user_preferences = {}  # Store user preferences for personalization

# Enhanced responses with more depth and intelligence
enhanced_responses = [
    "That's an intriguing question. Based on my analysis, I can provide several perspectives on this topic.",
    "I've analyzed your query carefully. The most comprehensive answer involves considering multiple factors including: ",
    "This is a complex topic with several dimensions to consider. From my understanding, the key aspects are:",
    "Your question touches on an interesting area. After processing relevant information, I can offer this detailed explanation:",
    "I've synthesized information from multiple domains to address your query. Here's my comprehensive response:",
    "After examining this question from various angles, I believe the most accurate response would incorporate these insights:",
    "This is a nuanced topic. Let me break down the key components for you in a structured way:",
    "I've considered the latest research on this topic. The most current understanding suggests that:",
    "Your question requires considering both theoretical frameworks and practical applications. Here's my synthesis:",
    "I'm applying advanced reasoning to your question. From multiple analytical perspectives, here's what I can tell you:"
]

# Enhanced topic-specific responses with greater depth
topic_responses = {}

# Add domain-specific knowledge bases
domain_knowledge = {
    "technology": [
        "The latest advancements in AI include multimodal models that can process text, images, and audio simultaneously.",
        "Quantum computing is progressing rapidly with recent breakthroughs in error correction.",
        "Edge computing is becoming increasingly important for real-time applications with low latency requirements.",
        "Blockchain technology has evolved beyond cryptocurrencies to support various decentralized applications."
    ],
    "science": [
        "Recent discoveries in CRISPR gene editing have opened new possibilities for treating genetic disorders.",
        "Advances in materials science have led to the development of metamaterials with properties not found in nature.",
        "The field of exoplanet research has identified numerous potentially habitable planets outside our solar system.",
        "Neuroscience research continues to improve our understanding of brain plasticity and cognitive functions."
    ],
    "history": [
        "Historical analysis often requires considering multiple perspectives and primary sources.",
        "Archaeological techniques have evolved to include sophisticated dating methods and non-invasive scanning.",
        "Comparative historical analysis helps identify patterns across different civilizations and time periods.",
        "Digital humanities tools have revolutionized how historians access and analyze historical documents."
    ]
}

def analyze_sentiment(message):
    """Analyze the sentiment of a message to provide more appropriate responses"""
    # Simple sentiment analysis
    positive_words = ["good", "great", "excellent", "amazing", "awesome", "like", "love", "happy"]
    negative_words = ["bad", "terrible", "awful", "dislike", "hate", "sad", "angry", "frustrating"]
    
    message_lower = message.lower()
    positive_score = sum(word in message_lower for word in positive_words)
    negative_score = sum(word in message_lower for word in negative_words)
    
    if positive_score > negative_score:
        return "positive"
    elif negative_score > positive_score:
        return "negative"
    else:
        return "neutral"

def extract_entities(message):
    """Extract key entities and topics from the message"""
    # Simple keyword extraction
    entities = []
    
    # Check for common topics
    topics = ["AI", "machine learning", "data", "science", "technology", "history", 
              "art", "music", "programming", "coding", "health", "medicine"]
    
    for topic in topics:
        if topic.lower() in message.lower():
            entities.append(topic)
    
    return entities

def generate_contextual_response(message, conversation_history=None, entities=None):
    """Generate a response that considers conversation context and extracted entities"""
    if not conversation_history:
        conversation_history = []
    if not entities:
        entities = []
    
    # Check for questions
    is_question = "?" in message
    
    # Generate base response
    if is_question:
        base_response = random.choice(enhanced_responses)
    else:
        base_response = "I've processed your message carefully. "
    
    # Add context from conversation history
    if len(conversation_history) > 2:
        base_response += "Considering our previous conversation, "
    
    # Add entity-specific information
    for entity in entities:
        for domain, knowledge in domain_knowledge.items():
            if entity.lower() in domain.lower():
                base_response += f"{random.choice(knowledge)} "
    
    return base_response

def chat_response(message, image_data=None):
    try:
        # Store message in conversation history
        global conversation_history
        conversation_history.append({"role": "user", "content": message})
        if len(conversation_history) > 10:  # Keep only last 10 messages
            conversation_history = conversation_history[-10:]
        
        # Handle image if provided
        image_description = ""
        if image_data:
            try:
                # Enhanced image description
                image_description = "I've analyzed the image you shared. "
                image_description += random.choice([
                    "I can see the visual content and will incorporate it into my response.",
                    "The image provides helpful context for understanding your query.",
                    "I've processed the visual information from your image.",
                    "Thank you for sharing this visual content which helps clarify your question."
                ])
            except Exception as e:
                print(f"Error processing image: {e}")
                image_description = "I had trouble processing your image."
        
        # Analyze the message
        sentiment = analyze_sentiment(message)
        entities = extract_entities(message)
        
        # Check if the response should be topic-specific
        global current_topic
        if current_topic and current_topic in topic_responses and topic_responses[current_topic]:
            # Use a topic-specific response
            response = random.choice(topic_responses[current_topic])
            response = f"[{current_topic}] {response}"
        else:
            # Generate a more intelligent contextual response
            response = generate_contextual_response(message, conversation_history, entities)
        
        # Add personalization based on message content and sentiment
        if sentiment == "positive":
            response = "I appreciate your positive perspective. " + response
        elif sentiment == "negative":
            response = "I understand your concerns. " + response
        
        # Add specificity based on message content
        if "explain" in message.lower() or "how" in message.lower():
            response += " Let me explain this step by step: "
        elif "compare" in message.lower():
            response += " When comparing these concepts, it's important to consider multiple dimensions: "
        elif "recommend" in message.lower() or "suggest" in message.lower():
            response += " Based on the information available, here are my recommendations: "
        
        # Add image description if there was an image
        if image_description:
            response = image_description + " " + response
        
        # Store response in conversation history
        conversation_history.append({"role": "ai", "content": response})
            
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        traceback.print_exc()
        return "I'm having trouble processing your request right now. Could you please try again with a different query?"

def generate_image(prompt):
    try:
        # Create a matplotlib figure for the generated image
        plt.figure(figsize=(10, 8))
        
        # Analyze the prompt to determine the type of image to generate
        prompt_lower = prompt.lower()
        
        # Extract key themes from the prompt
        themes = []
        for word in ["abstract", "landscape", "portrait", "chart", "graph", "diagram", 
                     "data", "science", "art", "mountain", "ocean", "sky", "city", 
                     "building", "nature", "technology", "space", "conceptual"]:
            if word in prompt_lower:
                themes.append(word)
        
        # If no specific themes were found, determine a default theme
        if not themes:
            if any(word in prompt_lower for word in ["show", "visualize", "display", "data"]):
                themes.append("data")
            elif any(word in prompt_lower for word in ["beautiful", "pretty", "aesthetic"]):
                themes.append("art")
            elif any(word in prompt_lower for word in ["nature", "natural", "outdoor"]):
                themes.append("landscape")
            else:
                themes.append("abstract")  # Default to abstract
        
        # Convert prompt to a seed for reproducibility
        seed = sum(ord(c) for c in prompt)
        random.seed(seed)
        np.random.seed(seed)
        
        # Create a more sophisticated image based on the themes
        if "landscape" in themes:
            # Generate a more sophisticated landscape
            x = np.linspace(0, 12, 1000)
            # Create multiple layers for the landscape
            layers = 4
            colors = [(0.2, 0.7, 0.2, 0.6), (0.1, 0.5, 0.1, 0.7), 
                     (0.5, 0.7, 0.3, 0.8), (0.7, 0.8, 0.2, 0.9)]
            
            for i in range(layers):
                # Different wave pattern for each layer
                freq = 0.5 + i * 0.2
                amp = 1 + i * 0.5
                phase = i * np.pi / 4
                
                # Add some randomness
                noise = np.random.normal(0, 0.1, len(x))
                
                # Create the wave with some polynomial terms for variety
                y = amp * np.sin(freq * x + phase) + 0.1 * x + 0.05 * x**2 * np.sin(x/3) + noise
                
                # Normalize to place each layer correctly
                y = y - y.min() + i * 2
                
                # Plot the layer
                plt.fill_between(x, np.zeros_like(x) + i * 2, y, color=colors[i % len(colors)])
            
            # Add a sun or moon
            if "night" in prompt_lower or "moon" in prompt_lower:
                # Moon
                circle = plt.Circle((2, 8), 0.8, color=(0.9, 0.9, 0.95), alpha=0.8)
                plt.gca().add_patch(circle)
                # Stars
                stars_x = np.random.uniform(0, 12, 50)
                stars_y = np.random.uniform(6, 10, 50)
                plt.scatter(stars_x, stars_y, color='white', alpha=0.8, s=2)
            else:
                # Sun
                circle = plt.Circle((10, 8), 1.2, color=(1.0, 0.8, 0.2), alpha=0.7)
                plt.gca().add_patch(circle)
                
            plt.title(f"Generated Landscape: {prompt}")
            
        elif "portrait" in themes or "face" in prompt_lower:
            # Generate a more artistic portrait-like image
            grid_size = 40
            
            # Create base face shape
            x, y = np.meshgrid(np.linspace(-3, 3, grid_size), np.linspace(-4, 4, grid_size))
            z = np.exp(-(x**2 + y**2)/10)
            
            # Add features
            # Eyes
            z -= 0.3 * np.exp(-((x-1)**2 + (y-1)**2)/0.2)
            z -= 0.3 * np.exp(-((x+1)**2 + (y-1)**2)/0.2)
            
            # Mouth
            mouth_curve = 0.2 * np.exp(-((x)**2 + (y+1.5)**2)/0.6)
            if "smile" in prompt_lower or "happy" in prompt_lower:
                z -= mouth_curve  # Smiling mouth
            else:
                z += mouth_curve * 0.5  # Neutral/serious mouth
            
            # Add some artistic distortion
            z += 0.1 * np.sin(5*x) * np.cos(5*y)
            
            # Choose a color scheme based on the prompt
            if "colorful" in prompt_lower:
                cmap = 'viridis'
            elif any(word in prompt_lower for word in ["sad", "blue", "melancholy"]):
                cmap = 'Blues'
            elif any(word in prompt_lower for word in ["angry", "red", "passionate"]):
                cmap = 'Reds'
            elif any(word in prompt_lower for word in ["calm", "green", "peaceful"]):
                cmap = 'Greens'
            else:
                cmap = 'plasma'
                
            # Plot with sophisticated shading
            plt.imshow(z, cmap=cmap, extent=[-3, 3, -4, 4])
            plt.title(f"Generated Portrait: {prompt}")
            
        elif "chart" in themes or "graph" in themes or "data" in themes:
            # Generate a data visualization based on the prompt
            
            # Determine the type of chart to create
            if "bar" in prompt_lower or "comparison" in prompt_lower:
                # Create a bar chart
                categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
                values = np.random.rand(5) * 10 + 5
                
                # Make the data more meaningful
                if "growth" in prompt_lower:
                    values.sort()
                elif "decline" in prompt_lower:
                    values = np.sort(values)[::-1]
                
                plt.bar(categories, values, color=plt.cm.viridis(np.linspace(0, 1, 5)))
                plt.ylabel("Value")
                plt.title(f"Data Visualization: {prompt}")
                
            elif "line" in prompt_lower or "trend" in prompt_lower or "time" in prompt_lower:
                # Create a line chart
                x = np.linspace(0, 10, 100)
                
                # Generate different trend lines based on the prompt
                if "exponential" in prompt_lower or "rapid growth" in prompt_lower:
                    y = np.exp(x/5)
                elif "logarithmic" in prompt_lower:
                    y = np.log(x + 1)
                elif "cyclical" in prompt_lower or "seasonal" in prompt_lower:
                    y = 5 * np.sin(x) + x/2 + np.random.normal(0, 0.5, 100)
                elif "decline" in prompt_lower or "decrease" in prompt_lower:
                    y = 10 - x/2 + np.random.normal(0, 0.5, 100)
                else:
                    # Default growth with some randomness
                    y = x/2 + np.sin(x) + np.random.normal(0, 0.5, 100)
                
                plt.plot(x, y, linewidth=2)
                plt.fill_between(x, y, alpha=0.3)
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.title(f"Trend Analysis: {prompt}")
                
            elif "scatter" in prompt_lower or "correlation" in prompt_lower:
                # Create a scatter plot
                n_points = 100
                
                # Generate correlated or uncorrelated data based on the prompt
                if "positive correlation" in prompt_lower:
                    x = np.random.rand(n_points) * 10
                    y = x + np.random.normal(0, 1, n_points)
                    r = 0.9
                elif "negative correlation" in prompt_lower:
                    x = np.random.rand(n_points) * 10
                    y = 10 - x + np.random.normal(0, 1, n_points)
                    r = -0.9
                elif "no correlation" in prompt_lower:
                    x = np.random.rand(n_points) * 10
                    y = np.random.rand(n_points) * 10
                    r = 0
                else:
                    # Default: moderate positive correlation
                    x = np.random.rand(n_points) * 10
                    y = 0.5*x + np.random.normal(0, 2, n_points)
                    r = 0.5
                
                # Create scatter plot with a regression line
                plt.scatter(x, y, alpha=0.6)
                
                # Add regression line
                m, b = np.polyfit(x, y, 1)
                plt.plot(x, m*x + b, color='red', linestyle='--')
                
                plt.xlabel("Variable X")
                plt.ylabel("Variable Y")
                plt.title(f"Correlation Analysis (r≈{r:.2f}): {prompt}")
                
            else:
                # Default to a pie chart
                labels = ['Category A', 'Category B', 'Category C', 'Category D']
                sizes = np.random.rand(4)
                sizes = sizes / sizes.sum()
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                        shadow=True, startangle=90, colors=plt.cm.viridis(np.linspace(0, 1, 4)))
                plt.axis('equal')
                plt.title(f"Distribution Analysis: {prompt}")
                
        else:
            # Generate sophisticated abstract art
            # Create a complex function for the abstract art
            def complex_function(x, y, seed):
                random.seed(seed)
                terms = []
                # Add 3-5 random terms
                for _ in range(random.randint(3, 5)):
                    amplitude = random.uniform(0.1, 1.0)
                    freq_x = random.uniform(0.1, 5.0)
                    freq_y = random.uniform(0.1, 5.0)
                    phase_x = random.uniform(0, 2*np.pi)
                    phase_y = random.uniform(0, 2*np.pi)
                    
                    # Create various mathematical forms
                    form = random.choice(['sin', 'cos', 'exp', 'log'])
                    if form == 'sin':
                        term = amplitude * np.sin(freq_x*x + phase_x) * np.sin(freq_y*y + phase_y)
                    elif form == 'cos':
                        term = amplitude * np.cos(freq_x*x + phase_x) * np.cos(freq_y*y + phase_y)
                    elif form == 'exp':
                        term = amplitude * np.exp(-((x-phase_x)**2 + (y-phase_y)**2)/(freq_x+freq_y))
                    else:  # log
                        term = amplitude * np.log(np.sqrt((x-phase_x)**2 + (y-phase_y)**2) + 1)
                    
                    terms.append(term)
                
                # Combine all terms
                result = sum(terms)
                return result
            
            # Create data for visualization
            x = np.linspace(-5, 5, 1000)
            y = np.linspace(-5, 5, 1000)
            X, Y = np.meshgrid(x, y)
            Z = complex_function(X, Y, seed)
            
            # Determine a suitable color map based on the prompt
            if "warm" in prompt_lower or "fire" in prompt_lower:
                cmap = 'hot'
            elif "cool" in prompt_lower or "water" in prompt_lower or "ocean" in prompt_lower:
                cmap = 'ocean'
            elif "rainbow" in prompt_lower or "colorful" in prompt_lower:
                cmap = 'rainbow'
            elif "earth" in prompt_lower or "natural" in prompt_lower:
                cmap = 'terrain'
            else:
                # Choose a random interesting colormap
                cmap = random.choice(['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
            
            # Create the visualization
            plt.imshow(Z, extent=[-5, 5, -5, 5], origin='lower', cmap=cmap)
            plt.title(f"Generated Art: {prompt}")
        
        plt.axis('off')
        
        # Save the image to a BytesIO object
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', dpi=150)
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
    global training_progress, is_training, topic_responses, current_topic, topic_knowledge
    
    try:
        # Simulate training process
        is_training = True
        training_progress = 0
        
        # Check if this is custom topic training
        if dataset_ref == "custom_topic" and custom_topic:
            print(f"Training on custom topic: {custom_topic}")
            current_topic = custom_topic
            
            # Generate more sophisticated topic-specific facts and responses
            facts, responses = generate_enhanced_topic_facts(custom_topic)
            
            # Store the responses for later use
            if responses:
                topic_responses[custom_topic] = responses
                
            # Store the facts as knowledge
            if not custom_topic in topic_knowledge:
                topic_knowledge[custom_topic] = []
            
            topic_knowledge[custom_topic].extend([fact for fact in facts['fact'] if fact not in topic_knowledge[custom_topic]])
        
        # Enhanced simulation of the training process with more sophisticated knowledge acquisition
        for i in range(21):
            if not is_training:  # Allow cancellation
                break
                
            # More detailed training process
            if i < 5:
                # Data preparation and preprocessing
                print(f"Processing and analyzing data for {dataset_ref or custom_topic}...")
            elif i < 10:
                # Feature extraction and model initialization
                print(f"Extracting key concepts from {dataset_ref or custom_topic}...")
            elif i < 15:
                # Training iterations
                print(f"Building knowledge representation for {dataset_ref or custom_topic}...")
            else:
                # Fine-tuning and validation
                print(f"Refining understanding of {dataset_ref or custom_topic}...")
            
            # Update training progress
            training_progress = min(100, int(i * 5))
            time.sleep(0.5)  # Simulate processing time
            
        # Finalize training with knowledge integration
        if is_training:  # Make sure training wasn't cancelled
            print(f"Completed training on {dataset_ref or custom_topic}")
            training_progress = 100
            
            # Generate additional domain-specific knowledge
            if custom_topic:
                domain = None
                for d in domain_knowledge.keys():
                    if d.lower() in custom_topic.lower():
                        domain = d
                        break
                
                if domain:
                    # Extend domain knowledge
                    if len(domain_knowledge.get(domain, [])) < 10:  # Only add if we don't have too many entries already
                        new_facts = [
                            f"Recent studies on {custom_topic} have shown promising results in various applications.",
                            f"The field of {custom_topic} continues to evolve with new methodologies and techniques.",
                            f"Experts in {custom_topic} emphasize the importance of understanding fundamental principles."
                        ]
                        
                        if domain in domain_knowledge:
                            domain_knowledge[domain].extend(new_facts)
                        else:
                            domain_knowledge[domain] = new_facts
                else:
                    # Create a new domain if appropriate
                    if custom_topic not in domain_knowledge:
                        domain_knowledge[custom_topic] = [
                            f"I've learned that {custom_topic} has multiple aspects worth exploring.",
                            f"Understanding {custom_topic} requires considering various interconnected factors.",
                            f"When analyzing {custom_topic}, it's important to consider the broader context."
                        ]
                        
        is_training = False
        
        return True
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        is_training = False
        training_progress = 0
        return False

def generate_enhanced_topic_facts(topic):
    """Generate more sophisticated facts about a specific topic with better structure and depth"""
    try:
        # Framework for creating more sophisticated knowledge representation
        categories = ["fundamentals", "applications", "recent_developments", "key_concepts", "common_misconceptions"]
        
        # Create a more structured pandas DataFrame with categorized facts
        fact_data = {
            'topic': [],
            'category': [],
            'fact_id': [],
            'fact': [],
            'confidence': [],
            'source': []
        }
        
        fact_id = 1
        for category in categories:
            for _ in range(3):  # 3 facts per category
                fact_data['topic'].append(topic)
                fact_data['category'].append(category)
                fact_data['fact_id'].append(fact_id)
                
                # Generate category-specific facts
                if category == "fundamentals":
                    fact = f"One of the core principles of {topic} is the interconnection between theory and practice."
                    if fact_id % 3 == 0:
                        fact = f"At its foundation, {topic} builds upon established frameworks while introducing novel approaches."
                    elif fact_id % 3 == 1:
                        fact = f"The fundamental concepts in {topic} provide a structured approach to understanding the domain."
                elif category == "applications":
                    fact = f"A practical application of {topic} can be seen in solving complex real-world problems."
                    if fact_id % 3 == 0:
                        fact = f"{topic} has been successfully applied in various domains including education and industry."
                    elif fact_id % 3 == 1:
                        fact = f"Recent applications of {topic} demonstrate its versatility and practical value."
                elif category == "recent_developments":
                    fact = f"Recent research in {topic} has expanded our understanding of key methodologies."
                    if fact_id % 3 == 0:
                        fact = f"The last few years have seen significant advancements in how we approach {topic}."
                    elif fact_id % 3 == 1:
                        fact = f"Emerging trends in {topic} suggest a shift toward more integrated approaches."
                elif category == "key_concepts":
                    fact = f"Among the essential concepts in {topic} is the principle of continuous improvement."
                    if fact_id % 3 == 0:
                        fact = f"Understanding the relationship between different aspects of {topic} is crucial for mastery."
                    elif fact_id % 3 == 1:
                        fact = f"A key framework in {topic} involves systematic analysis and synthesis of information."
                else:  # misconceptions
                    fact = f"A common misconception about {topic} is that it requires extensive prior knowledge."
                    if fact_id % 3 == 0:
                        fact = f"Unlike popular belief, {topic} is accessible to beginners with proper guidance."
                    elif fact_id % 3 == 1:
                        fact = f"People often incorrectly assume that {topic} is primarily theoretical rather than practical."
                
                fact_data['fact'].append(fact)
                fact_data['confidence'].append(np.random.uniform(0.85, 0.99))
                fact_data['source'].append(f"Training dataset {fact_id}")
                
                fact_id += 1
        
        # Create DataFrame
        facts = pd.DataFrame(fact_data)
        
        # Generate sophisticated responses based on the categorized facts
        responses = []
        
        # Add category-specific responses
        for category in categories:
            category_facts = facts[facts['category'] == category]
            if not category_facts.empty:
                for _, row in category_facts.iterrows():
                    responses.append(f"Regarding {topic}, {row['fact']}")
        
        # Add synthesized responses that combine multiple facts
        responses.extend([
            f"In my analysis of {topic}, I've found that it encompasses multiple dimensions including theoretical frameworks and practical applications.",
            f"My training on {topic} has revealed several interconnected concepts that provide a comprehensive understanding of the subject.",
            f"When considering {topic}, it's important to recognize both established principles and emerging developments in the field.",
            f"Based on my knowledge of {topic}, I can offer insights that integrate fundamental concepts with recent advancements.",
            f"The study of {topic} benefits from a multidisciplinary approach that considers various perspectives and methodologies."
        ])
        
        return facts, responses
        
    except Exception as e:
        print(f"Error generating enhanced topic facts: {e}")
        traceback.print_exc()
        return pd.DataFrame(), []

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