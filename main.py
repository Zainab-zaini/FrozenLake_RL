
import os
import json
import random
import numpy as np
import gymnasium as gym

from agent import QLearningAgent
from utils import (
    save_training_plot,
    evaluate_policy,
    save_gif
)

SEED = 42
EPISODES = 5000
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


def train_agent(learning_rate=0.1, episodes=EPISODES):
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False
    )

    env.action_space.seed(SEED)

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=learning_rate,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.999
    )

    rewards = []

    for episode in range(episodes):
        state, _ = env.reset(seed=SEED + episode)

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            done = terminated or truncated

            agent.learn(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_exploration()
        rewards.append(total_reward)

    env.close()

    return agent, rewards


# Part 1: Train the final agent
agent, rewards = train_agent()

# Save the learned Q-table
np.save(
    os.path.join(OUTPUT_DIR, "q_table.npy"),
    agent.q_table
)

# Save the training graph
save_training_plot(
    rewards,
    os.path.join(OUTPUT_DIR, "training_curve.png")
)

# Part 2: Evaluate the final agent
success_rate, average_steps = evaluate_policy(
    agent.q_table,
    episodes=500
)

# Save a demonstration GIF
gif_reward, gif_steps = save_gif(
    agent.q_table,
    os.path.join(OUTPUT_DIR, "trained_agent.gif")
)

# Part 3: Hyperparameter sensitivity experiment
sensitivity = {}

for lr in [0.05, 0.1, 0.2]:
    if lr == 0.1:
        test_agent = agent
    else:
        test_agent, _ = train_agent(
            learning_rate=lr,
            episodes=EPISODES
        )

    rate, steps = evaluate_policy(
        test_agent.q_table,
        episodes=500
    )

    sensitivity[str(lr)] = {
        "success_rate": rate,
        "average_steps": steps
    }

# Plot learning-rate comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))

plt.bar(
    sensitivity.keys(),
    [v["success_rate"] * 100 for v in sensitivity.values()]
)

plt.xlabel("Learning Rate")
plt.ylabel("Success Rate (%)")
plt.title("Hyperparameter Sensitivity")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "sensitivity.png"),
    dpi=150
)

plt.close()

# Save measured results
results = {
    "environment": "FrozenLake-v1",
    "is_slippery": False,
    "algorithm": "Tabular Q-learning",
    "training_episodes": EPISODES,
    "evaluation_episodes": 500,
    "success_rate": success_rate,
    "average_steps": average_steps,
    "final_epsilon": agent.epsilon,
    "gif_reward": gif_reward,
    "gif_steps": gif_steps,
    "learning_rate_comparison": sensitivity,
    "seed": SEED
}

with open(
    os.path.join(OUTPUT_DIR, "results.json"),
    "w"
) as file:
    json.dump(results, file, indent=4)

print("\nTRAINING AND EVALUATION RESULTS")
print(json.dumps(results, indent=4))

print("\nAll output files saved in the outputs folder.")
