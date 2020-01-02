"""
Very basic Q-learning model to play frozen lake from the gym library.
No hidden layers, simple q-table update method and exponential explore/exploit
rate of decay.
"""

import numpy as np
import gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")

action_space = env.action_space.n
state_space = env.observation_space.n

q_table = np.zeros((state_space, action_space))

episodes = 10000
max_steps = 100

learning_rate = 0.1
discount_rate = 0.99 # gama


# exploration vs exploitation: epsilon greedy strategy
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

episode_rewards = []

# Qlearning algorithm
# everything for a single episode
for episode in range(episodes):
    state = env.reset()
    done = False

    current_reward = 0

    for step in range(max_steps):
        # everything for a single time step

        # explore or exploit this step?
        exploration_threshold = random.uniform(0,1)
        if exploration_threshold > exploration_rate:
            # exploit
            action = np.argmax(q_table[state,:])

        else:
            # explore
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        # print(reward)
        # Update Qtable
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        current_reward += reward

        if done == True:
            break

    # exponential exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    episode_rewards.append(current_reward)


print(episode_rewards)
# calculate and print the average reward per thousand episodes
rewards_per_thousand_eps = np.split(np.array(episode_rewards), episodes/1000)
count = 1000

# print("====== Avg. reward per thousand episodes ======\n")
# for r in rewards_per_thousand_eps:
#     print(f"{count}: {sum(r/1000)}")
#     count += 1000
#
# print(f"====== Updated Qtable ======\n{q_table}")


# visualising 3 episodes to see how the agent behaves.
for episode in range(3):
    state = env.reset()
    done = False
    print(f"====== Episode {episode+1} ======\n\n")
    time.sleep(2)

    for step in range(max_steps):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()

            if reward == 1:
                print("====== Agent reached the goal! ======\n")
                time.sleep(3)
            clear_output(wait=True)
            break
        state = new_state
env.close()

"""
Final notes:
To further optimize, adding a reward decay for each step might help
the agent move more directly to the goal. This would have to be tweaked to find
the optimal decay value.
"""
