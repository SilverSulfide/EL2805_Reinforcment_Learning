# Load packages

import numpy as np
import gym

from collections import namedtuple

import torch

torch.manual_seed(0)
np.random.seed(0)

import matplotlib.pyplot as plt
from tqdm import trange

import utils as ut
from DDPG_agent import RandomAgent

from DDPG_soft_updates import soft_updates


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


# Import and initialize Lunar Lander
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 330  # Number of episodes to run for training
discount_factor = 0.99  # Value of gamma
n_ep_running_average = 50  # Running average of 50 episodes
m = len(env.action_space.high)  # dimensionality of the action
dim_state = len(env.observation_space.high)

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# get buffer
buffer_len = 30000
buffer = ut.ExperienceReplayBuffer(maximum_length=buffer_len)
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

# ---- Fill up the buffer ----- #
# Buffer_init = buffer_len
buffer_init = 300
state = env.reset()
agent = RandomAgent(m)

for i in range(buffer_init):
    # select random action
    action = agent.forward(state)
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

# intialise NN
# device = 'cuda:0'
device = 'cpu'
batch_size = 64
DDPG = ut.DDPG(dim_state, device)

print("NN initalised")

# initalise noise adding

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
best_loss = 0

for i in EPISODES:

    # intialise sampler
    ou_noise = ut.OUNoise()

    # Reset enviroment data
    done = False

    state = env.reset()

    # intialise variables
    total_episode_reward = 0.
    t = 0

    temp_loss = []

    while not done:

        # Create state tensor, remember to use single precision (torch.float32)
        state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32).to(device)

        # obtain acation from actor network
        action = DDPG.actor_network.inference(state_tensor)

        # add noise to the action -> NUMPY ARRAY
        action = action.cpu().detach().numpy() + ou_noise.select()

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        # TODO: verify this works
        next_state, reward, done, _ = env.step(action[0])

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        t += 1

        # Update buffer
        exp = Experience(state, action[0], reward, next_state, done)
        buffer.append(exp)

        state = next_state

        # ---- Actual training ----
        # Sample batch
        states, actions, rewards, next_states, dones = buffer.sample_batch(n=batch_size)

        # Calculate targets
        # TODO: sort gradients
        states_tensor = torch.tensor([states], dtype=torch.float32).to(device)

        next_states = torch.tensor([next_states], dtype=torch.float32).to(device)

        target_actions = DDPG.actor_target_network.inference(next_states)

        targets = DDPG.critic_target_network.forward(next_states, target_actions)

        targets = discount_factor * targets

        reward_tensor = torch.tensor([rewards], requires_grad=False, dtype=torch.float32).transpose(0, 1).to(
            device).unsqueeze(dim=0)

        mask = torch.tensor([dones], requires_grad=False).transpose(0, 1).to(device).unsqueeze(dim=0)
        targets = reward_tensor + targets * (mask.float() - 1).abs()

        # Get rid of target gradients
        targets = targets.detach()

        # update the critic network
        actions_tensor = torch.tensor([actions], dtype=torch.float32).to(device)
        values = DDPG.critic_network.forward(states_tensor, actions_tensor)

        loss = DDPG.critic_loss(values, targets)

        DDPG.backward_critic(loss)

        temp_loss.append(loss.detach().item())

        # if C steps have passed
        if t % 2 == 0:
            # SGD for actor network
            actor_actions = DDPG.actor_network.forward(states_tensor)
            critic_values = DDPG.critic_network.forward(states_tensor, actor_actions)
            loss = DDPG.actor_loss(critic_values)

            DDPG.backward_actor(loss)

            # soft update
            DDPG.target_critic_network = soft_updates(DDPG.critic_network, DDPG.critic_target_network, 10 ** (-3))
            DDPG.target_actor_network = soft_updates(DDPG.actor_network, DDPG.actor_target_network, 10 ** (-3))

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - Mean loss: {:.1f}".format(
            i, total_episode_reward, t,
            avg_reward,
            running_average(episode_number_of_steps, n_ep_running_average)[-1], np.mean(temp_loss)))

    if avg_reward > 125 and avg_reward > best_loss:
        best_loss = avg_reward
        print(best_loss)
        torch.save(DDPG.critic_network.state_dict(), 'critic_checkpoint.pth')
        torch.save(DDPG.actor_network.state_dict(), 'actor_checkpoint.pth')

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
