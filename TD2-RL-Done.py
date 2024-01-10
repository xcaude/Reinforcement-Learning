import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    Update the Q function for a given pair of action-state using Q-learning.

    Parameters:
    Q (numpy.ndarray): The Q-function table.
    s (int): Current state.
    a (int): Chosen action in the current state.
    r (float): Reward received after taking action a in state s.
    sprime (int): Next state after taking action a.
    alpha (float): Learning rate.
    gamma (float): Discount factor.

    Returns:
    numpy.ndarray: Updated Q-function table.
    """
    # Calculate the Q-value for the current state-action pair using the Q-learning formula
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[sprime, :]) - Q[s, a])
    return Q

def epsilon_greedy(Q, s, epsilon):
    """
    Select an action using the epsilon-greedy strategy based on the Q-function.

    Parameters:
    Q (numpy.ndarray): The Q-function table.
    s (int): Current state.
    epsilon (float): Exploration rate (0 for pure exploitation, 1 for pure exploration).

    Returns:
    int: The selected action.
    """
    if random.uniform(0, 1) < epsilon:
        # Explore: Choose a random action with probability epsilon
        action = random.randint(0, Q.shape[1] - 1)
    else:
        # Exploit: Choose the action with the highest Q-value for the current state
        action = np.argmax(Q[s, :])
    return action

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.05  # Set your own learning rate
    gamma = 0.9  # Set your own discount factor
    epsilon = 0.2  # Set your own exploration rate

    n_epochs = 100  # Set the number of training epochs
    max_itr_per_epoch = 30  # Set the maximum number of iterations per epoch
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )
            S = Sprime
            if done:
                break  # Exit loop if the episode is done

        print("episode #", e, " : r = ", r)
        rewards.append(r)

    print("Average reward = ", np.mean(rewards))

    # Plot the rewards in function of epochs
    plt.plot(rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Rewards Over Epochs")
    plt.show()

    print("Training finished.\n")

    """
    Evaluate the Q-learning algorithm
    """

    env.close()
