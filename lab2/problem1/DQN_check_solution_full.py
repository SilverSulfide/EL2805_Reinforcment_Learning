# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# NOTE: MODIFIED TO WORK WITH DQN CLASS FROM UTILS.PY

# Load packages
import numpy as np
import gym
import torch
from tqdm import trange

import matplotlib.pyplot as plt

import utils as ut


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def check_solution(path, device):
    # Import and initialize Mountain Car Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    # Load model
    NN = torch.load('neural-network-1.pth', map_location=torch.device(device))

    # Create DQN class
    n_actions = env.action_space.n  # Number of available actions
    dim_state = len(env.observation_space.high)  # State dimensionality
    hidden_dimension = 64
    DQN = ut.DQN(ut.net_builder, dim_state, n_actions, hidden_dimension, device)

    # Assign the loaded model to the DQN class
    DQN.network = NN

    # Parameters
    N_EPISODES = 50  # Number of episodes to run for trainings
    CONFIDENCE_PASS = 50

    # Reward
    episode_reward_list = []  # Used to store episodes reward

    # ---- FOr plotting episodic reward -----
    # list of episodes
    I = []

    # Reward
    episode_reward_list_random_agent = []  # Used to store episodes reward (random agent)

    # Simulate episodes
    print('Checking solution...')
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    for i in EPISODES:
        I.append(i)
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        while not done:
            # env.render()
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            q_values = DQN.forward(torch.tensor([state]).to(device))
            action = q_values.max(1)[1].item()
            next_state, reward, done, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()

    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)

    print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
        avg_reward,
        confidence))

    if avg_reward - confidence >= CONFIDENCE_PASS:
        print('Your policy passed the test!')
    else:
        print(
            "Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% "
            "confidence".format(
                CONFIDENCE_PASS))

    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        while not done:
            # Run random agent
            stupid_action = np.random.randint(0, n_actions)

            next_state, reward, done, _ = env.step(stupid_action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state

        # Append episode reward
        episode_reward_list_random_agent.append(total_episode_reward)

        # Close environment
        env.close()

    plt.plot(I, episode_reward_list, label="Our model")
    plt.plot(I, episode_reward_list_random_agent, label="Random model")
    plt.xlabel('Episodes')
    plt.ylabel('Reward for episode')
    plt.legend(loc="lower left")
    plt.show()

    # ---- Plot the max Q value ----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = np.arange(0, 1.5, 0.1).tolist()
    ys = np.arange(-np.pi, np.pi, 0.2).tolist()
    Y = []
    W = []
    Z = []

    def fun(x, y, net):
        state = np.array([0, x, 0, 0, y, 0, 0, 0])
        Q = net.forward(torch.tensor([state], dtype=torch.float32).to(device))
        val = Q.max(1)[0].item()
        return val

    # Create the triplets for plotting
    for i in range(len(xs)):
        for j in range(len(ys)):
            Y.append(xs[i])
            W.append(ys[j])
            Z.append(fun(xs[i], ys[j], DQN))
    ax.scatter(Y, W, Z)
    ax.set_xlabel('y')
    ax.set_ylabel('w')
    ax.set_zlabel('max Q value')
    plt.show()

    # ---- Plot the best action ----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = np.arange(0, 1.5, 0.1).tolist()
    ys = np.arange(-np.pi, np.pi, 0.2).tolist()
    Y = []
    W = []
    Z = []

    def fun(x, y, net):
        state = np.array([0, x, 0, 0, y, 0, 0, 0])
        Q = net.forward(torch.tensor([state], dtype=torch.float32).to(device))
        # print(Q)
        val = Q.max(1)[1].item()
        # print(val)
        return val

    # Create the triplets for plotting
    for i in range(len(xs)):
        for j in range(len(ys)):
            Y.append(xs[i])
            W.append(ys[j])
            Z.append(fun(xs[i], ys[j], DQN))
    ax.scatter(Y, W, Z)
    ax.set_xlabel('y')
    ax.set_ylabel('w')
    ax.set_zlabel('Best action')
    plt.show()
