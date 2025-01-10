import numpy as np
import random
import pygame
import pandas as pd
import matplotlib.pyplot as plt

# Define constants
GRID_SIZE = 5
GOAL = (4, 4)
OBSTACLE = [(1, 1), (2, 2), (3, 3)]  # You can add more obstacles here
REWARD_GOAL = 10
REWARD_MOVE = -1
REWARD_INVALID = -10
EPSILON = 0.1  # Exploration rate
ALPHA = 0.1   # Learning rate
GAMMA = 0.9   # Discount factor
EPISODES = 400

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption('Pathfinding Simulation')

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Create a 5x5 Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # 4 actions (up, down, left, right)

# Define actions: up, down, left, right
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # (y, x) changes

# Agent class
class Agent:
    def __init__(self, position):
        self.position = position
    
    def move(self, action):
        new_pos = (self.position[0] + actions[action][0], self.position[1] + actions[action][1])
        # Check for grid boundaries and obstacles
        if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE and new_pos not in OBSTACLE:
            self.position = new_pos
        return self.position

# Initialize agent
agent = Agent(position=(0, 0))

# Function to choose an action (epsilon-greedy)
def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice([0, 1, 2, 3])  # Random action (exploration)
    else:
        return np.argmax(q_table[state[0], state[1]])  # Best action (exploitation)

# Function to update Q-table
def update_q_table(state, action, reward, next_state):
    max_future_q = np.max(q_table[next_state[0], next_state[1]])  # Max Q value for the next state
    current_q = q_table[state[0], state[1], action]  # Current Q value for the action taken
    
    # Update Q value using the Q-learning rule
    q_table[state[0], state[1], action] = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)

# Function to visualize the environment
def draw_grid():
    screen.fill(WHITE)
    cell_size = 400 // GRID_SIZE

    # Draw grid
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, BLACK, rect, 1)
            if (x, y) == GOAL:
                pygame.draw.rect(screen, GREEN, rect)
            elif (x, y) in OBSTACLE:
                pygame.draw.rect(screen, RED, rect)
    
    # Draw the agent (car)
    agent_rect = pygame.Rect(agent.position[1] * cell_size, agent.position[0] * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, BLUE, agent_rect)

    pygame.display.update()

# Store rewards for graphing
rewards_list = []

# Main loop for training
for episode in range(EPISODES):
    agent.position = (0, 0)  # Reset agent position for each episode
    steps = 0
    total_reward = 0
    
    while agent.position != GOAL:
        for event in pygame.event.get():  # Handle events for smooth window interaction
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        state = agent.position
        action = choose_action(state)
        next_state = agent.move(action)
        
        # Check if the move is valid
        if next_state == GOAL:
            reward = REWARD_GOAL
        elif next_state not in OBSTACLE:
            reward = REWARD_MOVE
        else:
            reward = REWARD_INVALID
        
        update_q_table(state, action, reward, next_state)
        
        total_reward += reward
        steps += 1
        draw_grid()  # Update the visual
        pygame.time.delay(50)  # Slow down for visualization purposes
        
    print(f"Episode {episode}: Goal reached in {steps} steps, Total reward: {total_reward}")
    rewards_list.append(total_reward)

pygame.quit()

# After training, plot the rewards graph
plt.plot(rewards_list)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode')
plt.ylim(min(rewards_list), max(rewards_list)) 
plt.show()

# Saving Q-table to Excel
q_table_flattened = q_table.reshape(-1, 4)  # Flatten the Q-table for saving
df = pd.DataFrame(q_table_flattened, columns=["Up", "Down", "Left", "Right"])
df.to_excel(    "q_table_results.xlsx", index=False)

print("Q-table has been saved to 'q_table_results.xlsx'.")