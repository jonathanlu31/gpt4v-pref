import torch
from reward_predictor import RewardPredictorEnsemble

from pref_db import PrefDB, Segment

import wandb

wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="reward_model_walker_split_ai",
    # Track hyperparameters and run metadata
)

reward_model = RewardPredictorEnsemble(
    1, (17,), False, (6,), 1e-4, 32, "walker_ai_reward_split.pkl"
)
# reward_model.load_state_dict(torch.load("walker_ai_reward_split.pkl"))

train_db = PrefDB.load("train_preferences.pkl")
val_db = PrefDB.load("val_preferences.pkl")

Segment.set_include_action(False)

for i in range(60):
    stats = reward_model.train_one_epoch(train_db, val_db)
    wandb.log(stats)

# reward_model = RewardPredictorEnsemble(1, (4,), (1,), 3e-6, 32, 'cartpole_human_reward.pkl')
# reward_model.load_state_dict(torch.load('cartpole_human_reward.pkl'))

# print(reward_model.val_step(val_db))
