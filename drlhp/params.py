from argparse import ArgumentParser, Namespace


def parse_args(args) -> Namespace:
    parser = ArgumentParser()

    add_general_args(parser)
    add_reward_predictor_args(parser)
    return parser.parse_args(args)


def add_reward_predictor_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--include_actions",
        action="store_true",
        help="Flag to include actions in the reward predictions or just the observations",
        default="false",
    )


def add_general_args(parser: ArgumentParser):
    parser.add_argument("--seed", type=int, default=42)
