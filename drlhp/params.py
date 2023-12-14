from argparse import ArgumentParser, Namespace


def parse_args(args) -> Namespace:
    parser = ArgumentParser()

    add_general_args(parser)
    add_reward_predictor_args(parser)
    add_policy_args(parser)
    add_preference_arguments(parser)

    return parser.parse_args(args)


def add_reward_predictor_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="Flag to include actions in the reward predictions or just the observations",
        default=False,
    )
    parser.add_argument(
        "--reward-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the reward predictor",
    )
    parser.add_argument(
        "--rwd-mdl-bs",
        type=int,
        default=32,
        help="Batch size for training of the reward ensemble",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Number of reward predictors in the ensemble",
    )
    parser.add_argument(
        "--reward-model-checkpoint-path",
        type=str,
        default="reward_model.pkl",
        help="Path to save the best reward model",
    )
    parser.add_argument(
        "--num-reward-epochs-per-epoch",
        type=int,
        default=4,
        help="Number of times the reward training runs through the preference db every full training loop",
    )
    parser.add_argument(
        "--pretrained-reward-model-path",
        type=str,
        default="cheetah_reward.pkl",
        help="Pretrained reward model path",
    )


def add_policy_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--policy-learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for the policy agent",
    )
    parser.add_argument(
        "--train-steps-per-epoch",
        type=int,
        default=1e3,
        help="Number of training steps every epoch for the model.learn call",
    )
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--num-explore-steps", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1e6)
    parser.add_argument("--pretrained-agent-path", type=str, default='')


def add_preference_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--prefs-val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--collect-seg-interval",
        type=int,
        default=100,
        help="How often to collect segments along the policy rollouts",
    )
    parser.add_argument(
        "--seg-length",
        type=int,
        default=1,
        help="Length of each segment collected for ratin",
    )
    parser.add_argument("--max-prefs", type=int, default=500)
    parser.add_argument("--max-segs", type=int, default=100)


def add_general_args(parser: ArgumentParser):
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-env", type=str, default="HalfCheetah-v4")
