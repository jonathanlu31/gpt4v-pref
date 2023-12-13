import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader


class CoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, input: torch.Tensor):
        return self.net(input)


class RewardPredictorEnsemble(nn.Module):
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

    def __init__(
        self,
        num_predictors: int,
        ob_dim: int,
        include_actions: bool,
        ac_dim: int,
        lr: float,
        batch_size: int,
        checkpoint_path: str,
    ):
        super().__init__()
        self.num_predictors = num_predictors
        self.include_actions = include_actions
        if self.include_actions:
            input_dim = ob_dim[0] + ac_dim[0]
        else:
            input_dim = ob_dim[0]
        self.predictors = nn.ModuleList(
            [CoreNetwork(input_dim) for _ in range(num_predictors)]
        )
        self.bs = batch_size
        self.optimizer = torch.optim.AdamW(self.predictors.parameters(), lr=lr, weight_decay=6e-2)

        self.best_accuracy = 0
        self.checkpoint_path = checkpoint_path
        self.num_epochs = 0

    def forward(self, s1, s2):
        """Gets the reward preference

        Shape of sequences (batch_size x seq_len x (observation_dim + action_dim))
        Args:
            s1 (np.array): First trajectory segment to rate
            s2 (np.array): Second trajectory segment to rate

        predictor outputs: (batch, 2)
        Stacked: (num_predictors, batch, 2)

        Returns:
            torch.Tensor (2,): Summed logits of ratings from reward network
        """
        batch_size = s1.shape[0]
        seg_len = s1.shape[1]
        sum_s1 = torch.zeros(batch_size)
        sum_s2 = torch.zeros(batch_size)
        s1, s2 = s1.float(), s2.float()
        for pred in self.predictors:
            s1_reward = torch.sum(
                torch.vstack([pred(s1[:, i, :]) for i in range(seg_len)]).reshape(
                    (s1.shape[0], seg_len)
                ),
                dim=1,
            )
            s2_reward = torch.sum(
                torch.vstack([pred(s2[:, i, :]) for i in range(seg_len)]).reshape(
                    (s1.shape[0], seg_len)
                ),
                dim=1,
            )

            sum_s1 += s1_reward
            sum_s2 += s2_reward

        return (
            torch.hstack([sum_s1.unsqueeze(1), sum_s2.unsqueeze(1)])
            / self.num_predictors
        )

    def get_reward(self, observation, action):
        if self.include_actions:
            pred_input = torch.hstack([observation, action])
        else:
            pred_input = observation
        return torch.mean(
            torch.Tensor(
                [
                    predictor(pred_input)
                    for predictor in self.predictors
                ]
            )
        )

    def train_one_epoch(self, prefs_train, prefs_val):
        """
        Train all ensemble members for one epoch.
        """
        print(
            "Training/testing with %d/%d preferences"
            % (len(prefs_train), len(prefs_val))
        )
        if len(prefs_train) == 0:
            return

        total_steps = 0
        start_time = time.time()
        train_dataloader = DataLoader(prefs_train, self.bs, shuffle=True)
        total_train_loss = 0
        total_train_accuracy = 0
        for s1, s2, pref in train_dataloader:
            self.optimizer.zero_grad()
            network_pref = self(s1, s2)
            loss = F.cross_entropy(network_pref, pref)
            loss.backward()
            self.optimizer.step()
            total_train_loss += loss

            network_predictions = torch.argmax(network_pref, dim=1)

            mask = (pref == 0.5).any(dim=1)
            inverted_mask = ~mask
            labels = pref[inverted_mask]

            total_train_accuracy += torch.count_nonzero(
                network_predictions[inverted_mask] == torch.argmax(labels, dim=1)
            ) / len(labels)

            total_steps += 1

        result = self.val_step(prefs_val)
        self.num_epochs += 1
        end_time = time.time()
        rate = (total_steps) / (end_time - start_time)

        if result is None:
            return {
                "avg_train_loss": total_train_loss / total_steps,
                "avg_train_acc": total_train_accuracy / total_steps,
            }

        val_loss, val_acc = result

        return {
            "avg_train_loss": total_train_loss / total_steps,
            "avg_val_loss": val_loss,
            "avg_val_acc": val_acc,
            "avg_train_acc": total_train_accuracy / total_steps,
        }
        # easy_tf_log.tflog('reward_predictor_training_steps_per_second',
        #                   rate)

    def val_step(self, prefs_val):
        if len(prefs_val) == 0:
            return

        val_dataloader = DataLoader(prefs_val, self.bs, shuffle=True)

        running_loss = 0
        running_accuracy = 0
        with torch.no_grad():
            for s1, s2, pref in val_dataloader:
                network_pref = self(s1, s2)
                loss = F.cross_entropy(network_pref, pref)

                network_predictions = torch.argmax(network_pref, dim=1)

                mask = (pref == 0.5).any(dim=1)
                inverted_mask = ~mask
                labels = pref[inverted_mask]

                accuracy = torch.count_nonzero(
                    network_predictions[inverted_mask] == torch.argmax(labels, dim=1)
                ) / len(labels)

                running_loss += loss
                running_accuracy += accuracy

                # log validation loss
                # log accuracy
                # TODO: checkpoint model if it's better than previous accuracy
                if accuracy > self.best_accuracy:
                    torch.save(self.state_dict(), self.checkpoint_path)
                    self.best_accuracy = accuracy

        return running_loss / len(val_dataloader), running_accuracy / len(
            val_dataloader
        )
