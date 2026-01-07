# Lunar Lander with Deep Q-Learning (DQN)

This repository contains a PyTorch implementation of a Deep Q-Network (DQN) to solve the `LunarLander-v3` environment from the Farama Foundation's Gymnasium library.

This project was developed as a mid-term assignment focusing on Week 2: Deep Reinforcement Learning concepts.

---

## 🚀 Project Goal

The goal of the agent is to land the spaceship safely between the yellow flags.

- **Success Threshold:** An average score of 200+.
- **Input:** 8-dimensional continuous state vector (Coordinates, Velocity, Angle, Leg Sensors).
- **Output:** Discrete action space (Do Nothing, Fire Left, Fire Main, Fire Right).

---

## 📂 File Structure

- `lunar_lander_dqn.py`: The main Python script implementing the DQN algorithm.
- `lunar_lander_dqn.pth`: The trained model weights (The "Brain" of the agent).
- `README.md`: Project documentation.

---

## ⚙️ Installation

To run this project, you need Python installed along with the Gymnasium environment and PyTorch.

```bash
# Install the required libraries
pip install "gymnasium[box2d]" torch numpy
```
> **Note:** On some systems, you may need to install `swig` before installing `gymnasium`.

---

## 🏃‍♂️ How to Run

### 1. Training the Agent

By default, the script is set to train a new model. Run the file to start the training loop:

```bash
python lunar_lander_dqn.py
```

The training will run for 2000 episodes. It saves the model (overwriting the `.pth` file) whenever it achieves a new "All-Time High" score.

### 2. Watching the Agent (Test Mode)

To visualize the trained agent:

1.  Open `lunar_lander_dqn.py`.
2.  Scroll to the bottom `if __name__ == "__main__":` block.
3.  Comment out `train()` and uncomment `watch()`.

    ```python
    if __name__ == "__main__":
        # train()
        watch()  # <--- Uncomment this to see the agent fly
    ```
4.  Run the script again.

---

## 🧠 Algorithm Details

### Network Architecture

We use a fully connected neural network (Multi-Layer Perceptron) to approximate the Q-value function.

- **Input Layer:** 8 neurons (State dimension)
- **Hidden Layer 1:** 64 neurons (ReLU activation)
- **Hidden Layer 2:** 64 neurons (ReLU activation)
- **Output Layer:** 4 neurons (Action dimension)

### Hyperparameters

The following settings were used to achieve the final high score.

| Parameter         | Value  | Description                                      |
|-------------------|--------|--------------------------------------------------|
| Batch Size        | 64     | Number of transitions sampled from Replay Buffer |
| Gamma             | 0.99   | Discount factor for future rewards               |
| Learning Rate     | 5e-4   | Optimizer Step size (Adam)                       |
| Epsilon Start     | 1.0    | Initial exploration rate (100% random)           |
| Epsilon End       | 0.01   | Minimum exploration rate (1% random)             |
| Epsilon Decay     | 0.999  | Slow decay to allow maximum exploration          |
| Training Episodes | 2000   | Extended training for policy convergence         |
| Target Update     | 10     | Frequency of updating the Target Network         |

### Key Concepts Implemented

- **Experience Replay:** We store past experiences (`state`, `action`, `reward`, `next_state`) in a buffer to break correlations between consecutive samples, stabilizing training.
- **Target Network:** A separate network is used to calculate the target Q-values. It is updated every 10 episodes to prevent the "moving target" problem, ensuring the mathematical values don't spiral out of control.

---

## 📈 Results

The final model achieved exceptional performance after extended training.

- **Peak Score Achieved:** 298.16 (Near-Perfect Landing)
- **Consistency:** The agent consistently scores between 260-290 in test runs.

---

## 📚 Acknowledgments

- Course Materials: Week 2 (Deep Q-Networks).
- Based on concepts from Sutton & Barto, Chapter 13.

- Environment provided by Farama Foundation Gymnasium.
