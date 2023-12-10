from argparse import ArgumentParser, Namespace


def parse_args(args) -> Namespace:
    parser = ArgumentParser()

    add_general_args(parser)
    add_reward_predictor_args(parser)
    add_policy_args(parser)

    return parser.parse_args(args)


def add_reward_predictor_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--include_actions",
        action="store_true",
        help="Flag to include actions in the reward predictions or just the observations",
        default="false",
    )
    parser.add_argument(
        "--reward-learning-rate",
        type=float,
        default=1e-3,
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


def add_policy_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--policy-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the policy agent",
    )
    parser.add_argument(
        "--train_steps_per_epoch",
        type=int,
        default=1e6,
        help="Number of training steps every epoch for the model.learn call",
    )


def add_general_args(parser: ArgumentParser):
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prefs-val-fraction", type=float, default=0.2)
    parser.add_argument("--collect-seg-interval", type=int, default=50)
    parser.add_argument("--max-prefs", type=int, default=5)
    parser.add_argument("--max-segs", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=1)
