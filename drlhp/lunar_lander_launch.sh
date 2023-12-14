export PYTORCH_ENABLE_MPS_FALLBACK=1

/Users/brywong/opt/anaconda3/envs/285_final_project/bin/python /Users/brywong/Desktop/UCB/fa23/CS285/285-project/drlhp/run.py --base-env LunarLander-v2 --reward-model-checkpoint-path lunar_lander_human_reward.pkl --pretrained-reward-model-path "" --num-epochs 20 --collect-seg-interval 5
