import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# --- CONFIGURATION ---
BATCH_SIZE = 64         
GAMMA = 0.99            
EPSILON_START = 1.0     
EPSILON_END = 0.01      
EPSILON_DECAY = 0.999   # Decays per Episode now (Slower = Better)
LR = 5e-4               
MEMORY_SIZE = 100000    
TARGET_UPDATE = 10      

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE) 
        self.epsilon = EPSILON_START

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        curr_Q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_Q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_Q = rewards + (GAMMA * next_Q * (1 - dones))

        loss = F.mse_loss(curr_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # DECAY MOVED FROM HERE TO TRAIN LOOP

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Replace your train() function with this robust version:
def train():
    env = gym.make("LunarLander-v3") 
    agent = Agent(state_dim=8, action_dim=4)
    
    episodes = 2000
    best_score = 200  # Only save if we beat 200
    
    print("Training Started... (Running full 600 episodes to find the best model)")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
        
        # Decay Epsilon
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
            
        print(f"Episode: {episode}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        # SAVE STRATEGY: Only save if this is the best score we've ever seen
        if total_reward > best_score:
            best_score = total_reward
            torch.save(agent.policy_net.state_dict(), "lunar_lander_dqn.pth")
            print(f"🌟 NEW HIGH SCORE! Model saved with score: {best_score:.2f}")

    print("Training Complete. The best model has been saved.")
    env.close()
# Replace your old watch() function with this one:
def watch():
    # render_mode="human" makes the window appear
    env = gym.make("LunarLander-v3", render_mode="human")
    agent = Agent(state_dim=8, action_dim=4)
    
    # Load the trained brain
    try:
        agent.policy_net.load_state_dict(torch.load("lunar_lander_dqn.pth"))
        agent.policy_net.eval()
        print("Model loaded successfully! Getting ready to fly...")
    except FileNotFoundError:
        print("No saved model found. You need to train first!")
        return

    # Let's fly 5 times so you can see it clearly
    for i in range(1, 6):
        print(f"\n--- Flight Demonstration #{i} ---")
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select the best action (No random Epsilon here!)
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action = agent.policy_net(state_t).argmax().item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # This line makes sure the window processes events (keeps it from freezing)
            env.render() 
            
        print(f"Flight #{i} Result: Score {total_reward:.2f}")
        if total_reward > 200:
            print(">> SUCCESS: Perfect Landing! 🚀")
        else:
            print(">> RESULT: Safe, but maybe not perfect.")

    print("\nDemonstration finished. Closing window.")
    env.close()

if __name__ == "__main__":
    # train() 
    watch()