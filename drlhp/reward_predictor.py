import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoreNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(3, 3), bias=False)
        self.conv2 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), bias=False)
        self.conv3 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.conv4 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)

        self.fc5 = nn.Linear(int(w * h * c), 64)  # see todo
        self.fc6 = nn.Linear(64, 1)

    def net_moving_dot_features(s, batchnorm, dropout, training, reuse):
        # Action taken at each time step is encoded in the observations by a2c.py.
        a = s[:, 0, 0, -1]
        a = torch.tensor(a, dtype=torch.float32) / 4.0

        xc, yc = get_dot_position(s)
        xc = torch.tensor(xc, dtype=torch.float32) / 83.0
        yc = torch.tensor(yc, dtype=torch.float32) / 83.0

        features = [a, xc, yc]
        x = torch.stack(features, dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x[:, 0]

        return x

    def net_cnn(s, batchnorm, dropout, training, reuse):
        x = s / 255.0
        # Page 15: (Atari)
        # "[The] input is fed through 4 convolutional layers of size 7x7, 5x5, 3x3,
        # and 3x3 with strides 3, 2, 1, 1, each having 16 filters, with leaky ReLU
        # nonlinearities (α = 0.01). This is followed by a fully connected layer of
        # size 64 and then a scalar output. All convolutional layers use batch norm
        # and dropout with α = 0.5 to prevent predictor overfitting"
        x = F.leaky_relu(self.conv1(x))
        x = F.dropout(F.normalize(x), p=dropout, training=self.training)

        x = F.leaky_relu(self.conv2(x))
        x = F.dropout(F.normalize(x), p=dropout, training=self.training)

        x = F.leaky_relu(self.conv3(x))
        x = F.dropout(F.normalize(x), p=dropout, training=self.training)

        x = F.leaky_relu(self.conv4(x))
        x = F.dropout(F.normalize(x), p=dropout, training=self.training)

        w, h, c = x.shape[1:]
        x = torch.reshape(x, [-1, int(w * h * c)])  # todo: figure out what this is

        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = x[:, 0]

        return x


class RewardPredictorNetwork(nn.Module):
    """
    Predict the reward that a human would assign to each frame of
    the input trajectory, trained using the human's preferences between
    pairs of trajectories.

    Network inputs:
    - s1/s2     Trajectory pairs
    - pref      Preferences between each pair of trajectories
    Network outputs:
    - r1/r2     Reward predicted for each frame
    - rs1/rs2   Reward summed over all frames for each trajectory
    - pred      Predicted preference
    """

    def __init__(self, core_network, dropout, batchnorm, lr):
        super().__init__()

        self.core_network = CoreNetwork()

        self.dropout = dropout
        self.batchnorm = batchnorm

        self.optimizer = torch.optim.Adam(
            self.core_network.parameters(), lr=learning_rate
        )

    def forward(self, s1, s2):
        r1 = self.core_network(s1, self.batchnorm, self.dropout)
        r2 = self.core_network(s2, self.batchnorm, self.dropout)

        sum_r1 = torch.sum(r1, dim=1)
        sum_r2 = torch.sum(r2, dim=1)

        pref_pred = torch.softmax(torch.stack([sum_r1, sum_r2], dim=1))

        return pref_pred

    def update(self, s1, s2, human_pref):
        self.optimizer.zero_grad()
        network_pref = self(s1, s2)
        loss = F.cross_entropy(network_pref, human_pref)

        loss.backward()
        self.optimizer.step()
