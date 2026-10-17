
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.999
    ):
        self.q_table = np.zeros((n_states, n_actions))

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state, training=True):
        # Explore: choose a random action
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])

        # Exploit: choose the best known action
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]

        if done:
            next_value = 0
        else:
            next_value = np.max(self.q_table[next_state])

        target = reward + self.gamma * next_value

        self.q_table[state, action] = (
            old_value + self.lr * (target - old_value)
        )

    def decay_exploration(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
