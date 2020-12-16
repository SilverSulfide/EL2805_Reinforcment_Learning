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

from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal


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
env.seed(400)

# Parameters
N_episodes = 1600  # Number of episodes to run for training
discount_factor = 0.99  # Value of gamma
epsilon = 0.2
M = 10
n_ep_running_average = 50  # Running average of 50 episodes
m = len(env.action_space.high)  # dimensionality of the action
dim_state = len(env.observation_space.high)

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# intialise NN
if torch.cuda.is_available():
    print("Using gpu")
    device = 'cuda:0'
else:
    print("I cannot afford GPU")
    device = 'cpu'
batch_size = 64
PPO = ut.PPO(dim_state, device)

print("NN initalised")

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
best_loss = 0
q = 0

# Initialise the buffer
buffer = ut.Buffer()

for i in EPISODES:

    # Reset enviroment data
    done = False

    state = env.reset()

    # intialise variables
    total_episode_reward = 0.
    t = 0

    critic_loss = []
    actor_loss = []

    while not done:
        # Create state tensor, remember to use single precision (torch.float32)
        state_tensor = torch.tensor([state], dtype=torch.float32).to(device)

        # obtain mean and variance from actor network
        mu, var = PPO.actor_network.inference(state_tensor)

        # obtain policy distribution
        var = torch.diag_embed(var)
        distribution = MultivariateNormal(mu, var)

        # sample the distribution to get action
        action = distribution.sample()

        # get the probabilility of picking the action given state

        action_prob = distribution.log_prob(action)

        action = action.cpu().numpy()[0]
        action_prob = action_prob.cpu().item()

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        t += 1

        # Update buffer
        buffer.states.append(state)
        buffer.actions.append(action)
        buffer.rewards.append(reward)
        buffer.prob_action.append(action_prob)

        state = next_state

        buffer_len = len(buffer.states)

    # precompute y_i
    y_i = []
    for memo in range(buffer_len):

        # grab f last rewards
        # f = buffer_len - memo
        # temp_buffer = buffer.rewards[-f:]

        # TODO: make sure this works
        if memo == 0:
            target = 0
            for n, rew in enumerate(buffer.rewards):
                target += rew * discount_factor ** n
            y_i.append(target)
        else:
            target = (y_i[-1] - buffer.rewards[memo - 1]) / discount_factor
            y_i.append(target)

    # convert lists to arrays
    y_i = torch.tensor([y_i]).float().to(device).unsqueeze(dim=-1)
    states = torch.tensor([buffer.states], requires_grad=True, dtype=torch.float32).to(device)
    action_prob = torch.tensor([buffer.prob_action]).float().to(device)

    # ---- Actual training ----
    for epoch in range(M):
        # forward loop the critic
        critic_values = PPO.critic_network.forward(states)

        # compute critic loss
        loss = PPO.critic_loss(critic_values, y_i)

        # perfrom backward pass
        PPO.backward_critic(loss)

        # calculate advantage esimation
        psi = y_i - critic_values

        psi = psi.detach().squeeze(dim=-1)

        # calculate new action prob
        new_mu, new_var = PPO.actor_network.forward(states)
        new_var = torch.diag_embed(new_var)

        loss2 = PPO.actor_loss(new_mu, new_var, action_prob, psi)

        PPO.backward_actor(loss2)

        critic_loss.append(loss.item())
        actor_loss.append(loss2.item())

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Clear buffer
    buffer.clear_buffer()

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - Critic loss: {:.1f} - Actor loss: {:.1f}".format(
            i, total_episode_reward, t,
            avg_reward,
            running_average(episode_number_of_steps, n_ep_running_average)[-1], np.mean(critic_loss),
            np.mean(actor_loss)))

    if avg_reward > 125 and avg_reward > best_loss:
        best_loss = avg_reward
        torch.save(PPO.critic_network.state_dict(), 'critic_checkpoint.pth')
        torch.save(PPO.actor_network.state_dict(), 'actor_checkpoint.pth')

    if avg_reward > 140:
        q += 1
        if q == 10:
            break

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
