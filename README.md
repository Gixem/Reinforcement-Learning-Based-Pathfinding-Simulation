# Pathfinding Simulation with Reinforcement Learning

## Project Description
This project is a pathfinding simulation using **Reinforcement Learning (RL)**, where an agent learns to navigate a 5x5 grid from a starting point to a target point. The agent's movements are guided by a reward system, which assigns positive rewards for reaching the target and negative penalties for other actions.

The goal of this project is to demonstrate how reinforcement learning (RL) algorithms can be applied to solve real-world problems, such as navigating a grid-based environment. The simulation also visualizes the learning process, updating the Q-table and showing how the agent's performance improves over time.

## Features
- **5x5 Grid Setup**: The agent starts at a designated point and must navigate the grid to reach the target point.
- **Reinforcement Learning Algorithm (Q-Learning)**: The agent uses Q-Learning as part of Reinforcement Learning to find the optimal path to the target.
- **Reward System**: 
  - +10 for reaching the target
  - -1 for each movement
  - Invalid moves (going outside the grid) are penalized.
- **Learning Visualization**: The progress of learning is visualized by plotting rewards and updating the Q-table after each episode.

## How It Works
1. **Environment Setup**: The grid is initialized, and the agent's starting and target points are defined.
2. **Action Space**: The agent can move in four directions: up, down, left, and right.
3. **Q-Learning**: The agent updates the Q-table based on the actions taken, the resulting state, and the rewards received.
4. **Learning Progress**: As the agent completes more episodes, it learns to choose the optimal actions to reach the target in fewer steps.

## Installation

1. Clone the repository:


git clone https://github.com/yourusername/Pathfinding-Simulation.git
cd Pathfinding-Simulation

2. Install the required dependencies:

pip install -r requirements.txt

3. Run the simulation:
   
python pathfinding_simulation.py

## Requirements
Python 3.x
numpy
matplotlib
pandas
Results
The simulation's output includes:
![Excel](https://github.com/user-attachments/assets/8646433c-eca7-4c9c-93a8-7031dc09b62a)

## Reward Graph: Displays the reward progression over episodes.

![enyeni rewards_grapd](https://github.com/user-attachments/assets/d76fd428-6a67-4f46-a904-6afc7ed5cd4c)

## Final Agent Path: Visualizes the agent's learned path from start to target.
