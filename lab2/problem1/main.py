import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import trange
from DQN_agent import RandomAgent

import torch.nn.functional as F

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


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 1000  # Number of episodes
discount_factor = 0.95  # Value of the discount factor
n_ep_running_average = 50  # Running average of 50 episodes
n_actions = env.action_space.n  # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
C = 150*3

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []  # this list contains the total reward per episode
episode_number_of_steps = []  # this list contains the number of steps per episode

# Random agent initialization
# agent = RandomAgent(n_actions)

# get buffer
buffer = ut.ExperienceReplayBuffer(maximum_length=30000)
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

# intialise NN
device = 'cuda:0'
batch_size = 64
hidden_dimension = 64
DQN = ut.DQN(ut.net_builder, dim_state, n_actions, hidden_dimension, device)
gamma = 0.99

# initialise epsilon greedy sampler
greedy_sampler = ut.EpsilonSample(int(0.9 * N_episodes), n_actions)

# ---- Fill up the buffer ----- #
buffer_init = 1000
state = env.reset()

best_loss = 0

for i in range(buffer_init):
    # select epsilon-greedy action
    action = np.random.randint(0, n_actions)

    # Get next state and reward.  The done variable
    # will be True if you reached the goal position,
    # False otherwise
    next_state, reward, done, _ = env.step(action)

    # Update state for next iteration
    state = next_state
    # sample updates
    exp = Experience(state, action, reward, next_state, done)
    buffer.append(exp)

    if done:
        state = env.reset()

print("Buffer initialised")

# ------ Training process ----- #
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

c = 0

for i in EPISODES:

    # Reset enviroment data
    done = False

    state = env.reset()

    # intialise variables
    total_episode_reward = 0.
    t = 0

    while not done:
        if best_loss > 50:
            env.render()

        # Create state tensor, remember to use single precision (torch.float32)
        state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32).to(device)

        # perform forward pass on train network
        values = DQN.forward(state_tensor)

        # select epsilon-greedy action
        action = greedy_sampler.select(i, values)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        t += 1
        c += 1

        # Update buffer
        exp = Experience(state, action, reward, next_state, done)
        buffer.append(exp)

        state = next_state

        # ---- Actual training ----
        # Sample batch
        states, actions, rewards, next_states, dones = buffer.sample_batch(n=batch_size)

        # Calculate targets
        states_tensor = torch.tensor([states], dtype=torch.float32).squeeze().to(device)

        next_states = torch.tensor([next_states], dtype=torch.float32).squeeze().to(device)

        targets = DQN.forward_target(next_states)

        targets = gamma * torch.max(targets, dim=1)[0].unsqueeze(dim=1)

        reward_tensor = torch.tensor([rewards], requires_grad=False, dtype=torch.float32).transpose(0, 1).to(device)

        mask = torch.tensor([dones], requires_grad=False).transpose(0, 1).to(device)

        targets = reward_tensor + targets * (mask.float() - 1).abs()

        # update the training network
        actions_tensor = torch.tensor([actions]).long().transpose(0, 1).to(device)
        values = DQN.forward(states_tensor).gather(1, actions_tensor)


        loss = DQN.loss(values, targets)

        DQN.backward(loss)

        # if C steps have passed
        if c % C == 0:
            # copy the weights to the target network
            DQN.copy()

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            avg_reward,
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    if avg_reward > 50 and avg_reward > best_loss:
        best_loss = avg_reward
        print(best_loss)
        torch.save(DQN.network.state_dict(), 'checkpoint.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes + 1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
