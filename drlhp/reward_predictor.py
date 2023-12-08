import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)


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

    def __init__(self, ob_dim, ac_dim, lr):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.core_network = CoreNetwork(ob_dim, ac_dim)
        self.epoch = 100

        # self.dropout = dropout
        # self.batchnorm = batchnorm

        self.optimizer = torch.optim.Adam(self.core_network.parameters(), lr=lr)

        self.timestep = 0

    def forward(self, s1, s2):
        """Gets the reward preference

        Shape of sequences (batch_size x seq_len x (observation_dim + action_dim))

        Args:
            s1 (np.array): First trajectory segment to rate
            s2 (np.array): Second trajectory segment to rate

        Returns:
            torch.Tensor: Summed logits of ratings from reward network
        """
        seg_len = s1.shape[1]
        r1 = torch.Tensor([self.core_network(s1[:, i, :]) for i in range(seg_len)])
        r2 = torch.Tensor([self.core_network(s2[:, i, :]) for i in range(seg_len)])

        sum_r1 = torch.sum(r1, dim=1)
        sum_r2 = torch.sum(r2, dim=1)

        return torch.stack([sum_r1, sum_r2], dim=1)

    def get_reward(self, observation, state):
        return self.core_network(torch.stack([observation, state], dim=1))

    def update(self, s1, s2, human_pref):
        self.optimizer.zero_grad()
        network_pref = self(s1, s2)
        loss = F.cross_entropy(network_pref, human_pref)

        loss.backward()
        self.optimizer.step()

        self.timestep += 1

        if self.timestep % self.epoch == 0:
            torch.save(self.core_network.state_dict(), PATH)
