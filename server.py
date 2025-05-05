import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import pymunk
import random
import time
import traceback
import sys
import signal

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

print("Starting AI Playground Server with NumPy and Pymunk")
print(f"NumPy version: {np.__version__}")
print(f"Pymunk version: {pymunk.version}")

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros((1, output_size))
    
    def forward(self, x):
        # Make sure input is correctly shaped
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Forward pass through the network
        self.hidden = np.tanh(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        self.output = np.tanh(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output
    
    def get_weights(self):
        # Return all weights and biases as a dictionary
        return {
            'w1': self.weights_input_hidden.tolist(),
            'b1': self.bias_hidden.tolist(),
            'w2': self.weights_hidden_output.tolist(),
            'b2': self.bias_output.tolist()
        }
    
    def set_weights(self, weights):
        # Set weights from a dictionary
        self.weights_input_hidden = np.array(weights['w1'])
        self.bias_hidden = np.array(weights['b1'])
        self.weights_hidden_output = np.array(weights['w2'])
        self.bias_output = np.array(weights['b2'])
    
    def mutate(self, mutation_rate=0.1):
        # Add random values to weights to create mutations
        if random.random() < mutation_rate:
            self.weights_input_hidden += np.random.randn(*self.weights_input_hidden.shape) * 0.1
        if random.random() < mutation_rate:
            self.bias_hidden += np.random.randn(*self.bias_hidden.shape) * 0.1
        if random.random() < mutation_rate:
            self.weights_hidden_output += np.random.randn(*self.weights_hidden_output.shape) * 0.1
        if random.random() < mutation_rate:
            self.bias_output += np.random.randn(*self.bias_output.shape) * 0.1

class Walker:
    def __init__(self, space, input_size=14, hidden_size=8, output_size=4):
        # Neural network brain
        self.brain = NeuralNetwork(input_size, hidden_size, output_size)
        self.fitness = 0
        self.dead = False
        self.space = space
        self.bodies = []
        self.joints = []
        self.initial_x = 0
        self.age = 0  # Track how long the walker has been alive
        self.stale_counter = 0  # Track if walker is stuck
        self.last_position_x = 0
    
    def create_body(self, x, y):
        # Create the walker's physical body
        # Torso
        torso = pymunk.Body(mass=10, moment=pymunk.moment_for_box(10, (50, 80)))
        torso.position = (x, y)
        torso_shape = pymunk.Poly.create_box(torso, (50, 80))
        torso_shape.friction = 0.9
        
        # Head
        head = pymunk.Body(mass=2, moment=pymunk.moment_for_circle(2, 0, 20))
        head.position = (x, y - 60)
        head_shape = pymunk.Circle(head, 20)
        head_shape.friction = 0.9
        
        # Legs
        left_leg = pymunk.Body(mass=5, moment=pymunk.moment_for_box(5, (15, 60)))
        left_leg.position = (x - 20, y + 70)
        left_leg_shape = pymunk.Poly.create_box(left_leg, (15, 60))
        left_leg_shape.friction = 0.9
        
        right_leg = pymunk.Body(mass=5, moment=pymunk.moment_for_box(5, (15, 60)))
        right_leg.position = (x + 20, y + 70)
        right_leg_shape = pymunk.Poly.create_box(right_leg, (15, 60))
        right_leg_shape.friction = 0.9
        
        # Add bodies to space
        self.space.add(torso, torso_shape, head, head_shape, left_leg, left_leg_shape, right_leg, right_leg_shape)
        self.bodies = [torso, head, left_leg, right_leg]
        
        # Create joints
        neck_joint = pymunk.PivotJoint(torso, head, (0, -35), (0, 20))
        neck_joint.collide_bodies = False
        
        left_hip_joint = pymunk.PivotJoint(torso, left_leg, (-20, 35), (0, -30))
        left_hip_joint.collide_bodies = False
        
        right_hip_joint = pymunk.PivotJoint(torso, right_leg, (20, 35), (0, -30))
        right_hip_joint.collide_bodies = False
        
        # Add joints to space
        self.space.add(neck_joint, left_hip_joint, right_hip_joint)
        self.joints = [neck_joint, left_hip_joint, right_hip_joint]
        
        self.initial_x = x
        self.last_position_x = x
    
    def think(self):
        if self.dead:
            return
        
        try:
            # Gather inputs from body parts
            inputs = []
            for body in self.bodies:
                # Normalize x and y positions
                inputs.append(body.position.x / 800)
                inputs.append(body.position.y / 600)
                # Add rotation
                inputs.append(body.angle)
            
            # Get outputs from neural network
            outputs = self.brain.forward(np.array([inputs]))
            
            # Apply forces based on neural network outputs
            left_leg = self.bodies[2]
            right_leg = self.bodies[3]
            
            # Scale force by age to help initial stability
            force_scale = min(500, 200 + self.age * 10)
            
            left_leg.apply_force_at_local_point((outputs[0, 0] * force_scale, outputs[0, 1] * -force_scale), (0, 0))
            right_leg.apply_force_at_local_point((outputs[0, 2] * force_scale, outputs[0, 3] * -force_scale), (0, 0))
            
            # Update fitness based on distance traveled
            torso = self.bodies[0]
            distance_traveled = torso.position.x - self.initial_x
            self.fitness = max(0, distance_traveled)
            
            # Check if fallen or stuck
            if torso.position.y > 550:
                self.dead = True
                print(f"Walker died: fell down, fitness={self.fitness}")
            
            # Check if walker is stuck
            if abs(torso.position.x - self.last_position_x) < 0.5:
                self.stale_counter += 1
                if self.stale_counter > 120: # 2 seconds of being stuck
                    self.dead = True
                    print(f"Walker died: stuck, fitness={self.fitness}")
            else:
                self.stale_counter = 0
            
            self.last_position_x = torso.position.x
            self.age += 1
            
        except Exception as e:
            print(f"Error in walker thinking: {e}")
            traceback.print_exc()
            self.dead = True
    
    def remove_from_world(self):
        # Remove all bodies and joints from space
        try:
            for body in self.bodies:
                if body in self.space.bodies:
                    self.space.remove(body)
            for joint in self.joints:
                if joint in self.space.constraints:
                    self.space.remove(joint)
        except Exception as e:
            print(f"Error removing walker from world: {e}")
            traceback.print_exc()

class Population:
    def __init__(self, size=10):
        # Create physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)
        
        # Create ground
        self.ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.ground_shape = pymunk.Segment(self.ground_body, (0, 580), (800, 580), 5)
        self.ground_shape.friction = 1.0
        self.space.add(self.ground_body, self.ground_shape)
        
        # Create walkers
        self.walkers = []
        for _ in range(size):
            walker = Walker(self.space)
            walker.create_body(100, 450)
            self.walkers.append(walker)
        
        self.generation = 1
        self.best_fitness = 0
        self.generation_started = time.time()
        self.max_generation_time = 30  # Max 30 seconds per generation
    
    def run_step(self):
        # Have each walker think and apply forces
        for walker in self.walkers:
            if not walker.dead:
                walker.think()
        
        try:
            # Advance simulation with a fixed timestep
            self.space.step(1/60.0)
            
            # Check if all walkers are dead or timeout has been reached
            all_dead = all(walker.dead for walker in self.walkers)
            timeout = (time.time() - self.generation_started) > self.max_generation_time
            
            if all_dead or timeout:
                if timeout:
                    print(f"Generation {self.generation} timeout reached")
                self.natural_selection()
                return True  # New generation created
            return False  # Continuing with current generation
        except Exception as e:
            print(f"Error in simulation step: {e}")
            traceback.print_exc()
            # If we encounter an error, reset the simulation
            self.reset()
            return True
    
    def natural_selection(self):
        # Calculate maximum fitness
        max_fit = max((walker.fitness for walker in self.walkers), default=0)
        self.best_fitness = max(self.best_fitness, max_fit)
        
        print(f"Generation {self.generation} complete. Best fitness: {max_fit}")
        
        # Create mating pool based on fitness
        mating_pool = []
        for walker in self.walkers:
            # Normalized fitness between 0 and 1
            fitness_norm = walker.fitness / max_fit if max_fit > 0 else 0
            
            # Add copies to mating pool based on fitness
            n = int(fitness_norm * 100) + 1  # Ensure at least one copy
            mating_pool.extend([walker] * n)
        
        # Remove old walkers
        for walker in self.walkers:
            walker.remove_from_world()
        
        # Create new generation
        self.walkers = []
        for _ in range(10):  # Population size
            walker = Walker(self.space)
            
            # Select parent from mating pool if available
            if mating_pool:
                parent = random.choice(mating_pool)
                walker.brain.set_weights(parent.brain.get_weights())
                walker.brain.mutate(0.1)  # 10% mutation rate
            
            walker.create_body(100, 450)
            self.walkers.append(walker)
        
        self.generation += 1
        self.generation_started = time.time()
    
    def reset(self):
        # Clear all existing objects
        self.space.remove(self.space.bodies, self.space.shapes, self.space.constraints)
        
        # Create new space
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)
        
        # Create ground
        self.ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.ground_shape = pymunk.Segment(self.ground_body, (0, 580), (800, 580), 5)
        self.ground_shape.friction = 1.0
        self.space.add(self.ground_body, self.ground_shape)
        
        # Create walkers
        self.walkers = []
        for _ in range(10):
            walker = Walker(self.space)
            walker.create_body(100, 450)
            self.walkers.append(walker)
        
        self.generation = 1
        self.best_fitness = 0
        self.generation_started = time.time()

# Global population instance
population = None

@app.route('/api/init', methods=['POST'])
def initialize():
    global population
    try:
        if population is not None:
            # Clean up existing population
            for walker in population.walkers:
                walker.remove_from_world()
        
        population = Population(size=10)
        print("Simulation initialized")
        return jsonify({
            'status': 'success',
            'generation': population.generation,
            'best_fitness': population.best_fitness
        })
    except Exception as e:
        print(f"Error initializing simulation: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/step', methods=['GET'])
def step():
    global population
    try:
        if not population:
            return jsonify({'error': 'Population not initialized'}), 400
        
        # Run simulation step
        new_generation = population.run_step()
        
        # Get positions of all walkers for rendering
        walker_data = []
        for i, walker in enumerate(population.walkers):
            if not walker.dead:
                body_positions = []
                for body in walker.bodies:
                    body_positions.append({
                        'x': body.position.x,
                        'y': body.position.y,
                        'angle': body.angle
                    })
                walker_data.append({
                    'id': i,
                    'bodies': body_positions,
                    'fitness': walker.fitness
                })
        
        return jsonify({
            'status': 'success',
            'new_generation': new_generation,
            'generation': population.generation,
            'best_fitness': population.best_fitness,
            'walkers': walker_data
        })
    except Exception as e:
        print(f"Error in simulation step: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    global population
    try:
        if population:
            population.reset()
        else:
            population = Population(size=10)
        
        print("Simulation reset")
        return jsonify({
            'status': 'success',
            'generation': population.generation,
            'best_fitness': population.best_fitness
        })
    except Exception as e:
        print(f"Error resetting simulation: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Simulation server is running',
        'endpoints': ['/api/init', '/api/step', '/api/reset']
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'version': '1.0.0'
    })

# Handle graceful shutdown
def signal_handler(sig, frame):
    print('Shutting down server...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000, threaded=True) 