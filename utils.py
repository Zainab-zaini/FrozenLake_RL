
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio.v2 as imageio


def save_training_plot(rewards, output_path):
    plt.figure(figsize=(9, 5))

    plt.plot(rewards, alpha=0.3, label="Episode Reward")

    window = 100

    if len(rewards) >= window:
        moving_avg = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )

        plt.plot(
            np.arange(len(moving_avg)) + window - 1,
            moving_avg,
            label="100-Episode Moving Average"
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-Learning Training Progress")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_policy(q_table, episodes=500):
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False
    )

    successes = 0
    total_steps = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=1000 + episode)
        done = False
        steps = 0

        while not done and steps < 100:
            action = int(np.argmax(q_table[state]))

            state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            done = terminated or truncated
            steps += 1

            if reward > 0:
                successes += 1

        total_steps += steps

    env.close()

    success_rate = successes / episodes
    average_steps = total_steps / episodes

    return success_rate, average_steps


def save_gif(q_table, output_path):
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode="rgb_array"
    )

    state, _ = env.reset(seed=42)
    frames = [env.render()]
    done = False
    steps = 0
    reward = 0

    while not done and steps < 100:
        action = int(np.argmax(q_table[state]))

        state, reward, terminated, truncated, _ = (
            env.step(action)
        )

        frames.append(env.render())
        done = terminated or truncated
        steps += 1

    env.close()

    imageio.mimsave(
        output_path,
        frames,
        duration=0.5,
        loop=0
    )

    return reward, steps
