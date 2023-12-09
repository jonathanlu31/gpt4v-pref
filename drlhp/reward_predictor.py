import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader


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

    def __init__(self, ob_dim, ac_dim):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.core_network = CoreNetwork(ob_dim[0] + ac_dim[0])
        self.epoch = 100

        # self.dropout = dropout
        # self.batchnorm = batchnorm

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


class RewardPredictorEnsemble(nn.Module):
    def __init__(
        self, num_predictors: int, ob_dim: int, ac_dim: int, lr: float, batch_size: int
    ):
        super().__init__()
        self.predictors = nn.ModuleList(
            [RewardPredictorNetwork(ob_dim, ac_dim) for _ in range(num_predictors)]
        )
        self.n_steps = 0
        self.bs = batch_size
        self.optimizer = torch.optim.Adam(self.predictors.parameters(), lr=lr)

    def forward(self, s1, s2):
        return torch.mean([predictor(s1, s2) for predictor in self.predictors], axis=0)

    def get_reward(self, observation, state):
        return torch.mean(
            [predictor.get_reward(observation, state) for predictor in self.predictors]
        )

    def train_one_epoch(self, prefs_train, prefs_val, val_interval):
        """
        Train all ensemble members for one epoch.
        """
        print(
            "Training/testing with %d/%d preferences"
            % (len(prefs_train), len(prefs_val))
        )

        start_steps = self.n_steps
        start_time = time.time()
        train_dataloader = DataLoader(prefs_train, self.bs, shuffle=True)

        for s1, s2, pref in train_dataloader:
            self.optimizer.zero_grad()
            network_pref = self(s1, s2)
            loss = F.cross_entropy(network_pref, pref)

            loss.backward()
            self.optimizer.step()

            self.n_steps += 1

            if self.n_steps and self.n_steps % val_interval == 0:
                self.val_step(prefs_val)

        end_time = time.time()
        end_steps = self.n_steps
        rate = (end_steps - start_steps) / (end_time - start_time)
        # easy_tf_log.tflog('reward_predictor_training_steps_per_second',
        #                   rate)

    def val_step(self, prefs_val):
        val_dataloader = DataLoader(prefs_val, self.bs, shuffle=True)

        with torch.no_grad():
            for s1, s2, pref in val_dataloader:
                network_pref = self(s1, s2)
                loss = F.cross_entropy(network_pref, pref)

                network_predictions = torch.argmax(network_pref, dim=1)

                mask = (pref == 0.5).any(dim=1)
                inverted_mask = ~mask
                labels = pref[inverted_mask]

                accuracy = torch.count_nonzero(
                    network_predictions[inverted_mask] == labels
                ) / len(labels)

                # log validation loss
                # log accuracy
