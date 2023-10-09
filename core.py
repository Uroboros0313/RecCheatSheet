import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims=[64, 32, 1]):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))

        for i in range(1, len(hidden_dims)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
