import torch
import torch.nn as nn


class NNCritic(nn.Module):
    """ Create a single layer feedforward neural network """

    def __init__(self, input_size, output_size=1, hidden_size_1=400, hidden_size_2=200):
        super().__init__()

        self.layer_activation = nn.ReLU()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_size_1)

        # Create a hidden layer
        self.hidden_layer = nn.Linear(hidden_size_1, hidden_size_2)

        # Create output layer
        self.output_layer = nn.Linear(hidden_size_2, output_size)

    def forward(self, x, action):
        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.layer_activation(l1)

        # Compute second layer
        l2 = self.hidden_layer(l1)
        l2 = self.layer_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)

        return out


class NNActor(nn.Module):
    """ Create a single layer feedforward neural network """

    def __init__(self, input_size, output_size=2, hidden_size_1=400, hidden_size_2=200):
        super().__init__()

        self.layer_activation = nn.ReLU()
        self.mu_activation = nn.Tanh()
        self.var_activation = nn.Sigmoid()

        # Create input layer with ReLU activation
        self.shared_input_layer = nn.Linear(input_size, hidden_size_1)

        # Create a hidden layer
        self.hidden_layer = nn.Linear(hidden_size_1, hidden_size_2)

        # Create output layer
        self.output_layer = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.shared_input_layer(x)
        l1 = self.layer_activation(l1)


        # ---- First head ----
        # Compute second layer
        l2 = self.hidden_layer(l1)
        l2 = self.layer_activation(l2)

        # Compute output layer
        out1 = self.output_layer(l2)
        out1 = self.mu_activation(out1)


        # ---- Second head ----
        # Compute second layer
        l2_2 = self.hidden_layer(l1)
        l2_2 = self.layer_activation(l2_2)

        # Compute output layer
        out2 = self.output_layer(l2_2)
        out2 = self.var_activation(out2)

        return out1, out2

    # TODO: The Gaussian probably does not need to be here.

    @torch.no_grad()
    def inference(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.shared_input_layer(x)
        l1 = self.layer_activation(l1)


        # ---- First head ----
        # Compute second layer
        l2 = self.hidden_layer(l1)
        l2 = self.layer_activation(l2)

        # Compute output layer
        out1 = self.output_layer(l2)
        out1 = self.mu_activation(out1)


        # ---- Second head ----
        # Compute second layer
        l2_2 = self.hidden_layer(l1)
        l2_2 = self.layer_activation(l2_2)

        # Compute output layer
        out2 = self.output_layer(l2_2)
        out2 = self.var_activation(out2)

        return out1, out2
