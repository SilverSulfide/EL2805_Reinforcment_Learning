import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class EpsilonSample:
    def __init__(self, Z, n_actions, e_min=0.05, e_max=0.99):
        self.Z = Z
        self.e_min = e_min
        self.e_max = e_max
        self.n_actions = n_actions

    def select(self, k, values):
        # compute current epsilon
        e_k = max(self.e_min, self.e_max - (self.e_max - self.e_min) * k / (self.Z - 1))

        # roll dice

        if np.random.uniform() > e_k:
            return values.max(1)[1].item()

        else:
            return np.random.randint(0, self.n_actions)


class NN(nn.Module):
    """ Create a single layer feedforward neural network """

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.layer_activation = nn.ReLU()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Create a hidden layer
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)

        # Create output layer
        self.output_layer = nn.Linear(hidden_size, output_size)


def net_builder(input_size, output_size, hidden_size, device):
    return NN(input_size, output_size, hidden_size).to(device)


class DQN:
    """ Performs forward and backward passes for (target) network"""
    def __init__(self, net_builder, input_size, output_size, hidden_size, device):
        self.network = net_builder(input_size, output_size, hidden_size, device)
        self.target_network = net_builder(input_size, output_size, hidden_size, device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.network.input_layer(x)
        l1 = self.network.layer_activation(l1)

        # Compute second layer
        l2 = self.network.hidden_layer(l1)
        l2 = self.network.layer_activation(l2)

        # Compute output layer
        out = self.network.output_layer(l2)
        return out

    def loss(self, out, target):
        return F.mse_loss(out, target)

    def backward(self, loss):
        # reset gradients to 0
        self.optimizer.zero_grad()

        # get loss
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.)

        # Perform backward pass (backpropagation)
        self.optimizer.step()

    @torch.no_grad()
    def forward_target(self, x):

        # Compute first layer
        l1 = self.target_network.input_layer(x)
        l1 = self.target_network.layer_activation(l1)

        # Compute second layer
        l2 = self.target_network.hidden_layer(l1)
        l2 = self.target_network.layer_activation(l2)

        # Compute output layer
        out = self.target_network.output_layer(l2)
        return out

    def copy(self):
        """ Copies network parameters to target network"""
        for param_q, param_k in zip(self.network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize the evaluation net
            param_k.requires_grad = False  # do not update by gradient for evaluation net


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
