import torch
from reward_predictor import RewardPredictorEnsemble

from pref_db import PrefDB

import wandb

# wandb.login()

# run = wandb.init(
#     # Set the project where this run will be logged
#     project="reward_model",
#     # Track hyperparameters and run metadata
# )

# reward_model = RewardPredictorEnsemble(1, (17,), (6,), 5e-4, 32, 'cheetah_reward.pkl')

# train_db = PrefDB.load('train_preferences.pkl')
val_db = PrefDB.load('val_preferences.pkl')

# for i in range(70):
#     stats = reward_model.train_one_epoch(train_db, val_db)
#     wandb.log(stats)

reward_model = RewardPredictorEnsemble(1, (17,), (6,), 5e-4, 32, 'cheetah_reward.pkl')
reward_model.load_state_dict(torch.load('cheetah_reward.pkl'))

print(reward_model.val_step(val_db))
