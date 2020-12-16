import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import MultivariateNormal

from nets import NNCritic, NNActor


class PPO:
    def __init__(self, input_size, device):
        # FIXME: 2 outputs
        # https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440
        self.actor_network = NNActor(input_size).to(device)

        self.critic_network = NNCritic(input_size).to(device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=10 ** (-5))
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=10 ** (-3))

        self.epsilon = torch.tensor(0.2).to(device)

    def critic_loss(self, out, target):
        return F.mse_loss(out, target)

    # FIXME: implement
    def actor_loss(self, new_mu, new_var, old_probs, psi):

        # FIXME: sloppy device inheritance
        loss = torch.zeros(new_mu.shape).to(device=new_mu.device)

        for i in range(new_mu.shape[1]):
            distribution = MultivariateNormal(new_mu[:, i, :], new_var[:, i, ...])
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            r_theta = torch.exp(log_prob - old_probs)

            loss += torch.min(r_theta * psi[:, i, ...], self.epsilon_min(r_theta) * psi[:, i, ...])

        return - loss.mean()

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

    def epsilon_min(self, x):
        return torch.max(1 - self.epsilon, torch.min(x, 1 + self.epsilon))


class Buffer:
    # FIXME: lists might not be fastest
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.prob_action = []

    def clear_buffer(self):
        self.__init__()
