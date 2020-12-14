import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from nets import NNCritic, NNActor


class OUNoise:
    def __init__(self, sigma=0.04, mu=0.15):
        self.n = 0
        self.sigma = sigma * np.eye(2)
        self.mu = mu

    def select(self):
        # to make sure to start at t=0
        current = self.n

        n_t = -self.mu * self.n + np.random.multivariate_normal([0, 0], self.sigma)

        self.n = n_t

        return current


class DDPG:
    def __init__(self, input_size, device):
        self.actor_network = NNActor(input_size).to(device)
        self.actor_target_network = NNActor(input_size).to(device)

        self.critic_network = NNCritic(input_size).to(device)
        self.critic_target_network = NNCritic(input_size).to(device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=5*10**(-5))
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=5*10**(-4))

    def critic_loss(self, out, target):
        return F.mse_loss(out, target)

    # FIXME: dimensions
    def actor_loss(self, out):
        return -torch.mean(out)

    def backward_critic(self, loss):
        # reset gradients to 0
        self.critic_optimizer.zero_grad()

        # get loss
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=1.)

        # Perform backward pass (backpropagation)
        self.critic_optimizer.step()

    def backward_actor(self, loss):
        # reset gradients to 0
        self.actor_optimizer.zero_grad()

        # get loss
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=1.)

        # Perform backward pass (backpropagation)
        self.actor_optimizer.step()


class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """

    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)
