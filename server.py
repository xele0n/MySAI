import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import pymunk
import pymunk.pygame_util
import random
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros((1, output_size))
    
    def forward(self, x):
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
    
    def think(self):
        if self.dead:
            return
        
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
        
        left_leg.apply_force_at_local_point((outputs[0, 0] * 500, outputs[0, 1] * -500), (0, 0))
        right_leg.apply_force_at_local_point((outputs[0, 2] * 500, outputs[0, 3] * -500), (0, 0))
        
        # Update fitness based on distance traveled
        torso = self.bodies[0]
        distance_traveled = torso.position.x - self.initial_x
        self.fitness = max(0, distance_traveled)
        
        # Check if fallen or stuck
        if torso.position.y > 550 or self.fitness == 0:
            self.dead = True
    
    def remove_from_world(self):
        # Remove all bodies and joints from space
        for body in self.bodies:
            self.space.remove(body)
        for joint in self.joints:
            self.space.remove(joint)

class Population:
    def __init__(self, size=10):
        # Create physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)
        
        # Create ground
        ground = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(ground, (0, 580), (800, 580), 5)
        ground_shape.friction = 1.0
        self.space.add(ground, ground_shape)
        
        # Create walkers
        self.walkers = []
        for _ in range(size):
            walker = Walker(self.space)
            walker.create_body(100, 450)
            self.walkers.append(walker)
        
        self.generation = 1
        self.best_fitness = 0
    
    def run_step(self):
        # Have each walker think and apply forces
        for walker in self.walkers:
            if not walker.dead:
                walker.think()
        
        # Advance simulation
        self.space.step(1/60.0)
        
        # Check if all walkers are dead
        all_dead = all(walker.dead for walker in self.walkers)
        if all_dead:
            self.natural_selection()
            return True  # New generation created
        return False  # Continuing with current generation
    
    def natural_selection(self):
        # Calculate maximum fitness
        max_fit = max((walker.fitness for walker in self.walkers), default=0)
        self.best_fitness = max(self.best_fitness, max_fit)
        
        # Create mating pool based on fitness
        mating_pool = []
        for walker in self.walkers:
            # Normalized fitness between 0 and 1
            fitness_norm = walker.fitness / max_fit if max_fit > 0 else 0
            
            # Add copies to mating pool based on fitness
            n = int(fitness_norm * 100)
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

# Global population instance
population = None

@app.route('/api/init', methods=['POST'])
def initialize():
    global population
    population = Population(size=10)
    return jsonify({
        'status': 'success',
        'generation': population.generation,
        'best_fitness': population.best_fitness
    })

@app.route('/api/step', methods=['GET'])
def step():
    global population
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

@app.route('/api/reset', methods=['POST'])
def reset():
    global population
    population = Population(size=10)
    return jsonify({
        'status': 'success',
        'generation': population.generation,
        'best_fitness': population.best_fitness
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 