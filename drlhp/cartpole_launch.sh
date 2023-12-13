export PYTORCH_ENABLE_MPS_FALLBACK=1

/Users/lqi/miniconda3/envs/285-proj/bin/python /Users/lqi/cs285/proj/repo/drlhp/run.py --base-env CartPole-v1 --reward-model-checkpoint-path cartpole_human_reward.pkl --pretrained-reward-model-path cartpole_human_reward.pkl --num-epochs 20 --collect-seg-interval 5
