import logging
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import tqdm

import numpy as np
import random
import sys
from torch.multiprocessing import Process, Queue
import os
import copy

from custom_env import CustomEnv
from params import parse_args

from pref_interface import PrefInterface
from reward_predictor import RewardPredictorEnsemble
from pref_db import PrefDB, PrefBuffer, Segment


def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class PretrainCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        num_steps_explore,
        seg_pipe,
        collect_seg_interval,
        save_interval,
        verbose=1,
    ):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.num_steps_explore = num_steps_explore
        self.collect_seg_interval = collect_seg_interval
        self.seg_pipe = seg_pipe

        self.log_dir = os.path.dirname(__file__)

        self.save_interval = save_interval
        self.save_path = os.path.join(self.log_dir, "agent")
        self.best_mean_reward = -float("inf")

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        base_env = self.training_env.envs[0].env
        obs, info = base_env.reset()
        seg = Segment()

        for i in range(1, self.num_steps_explore + 1):
            ac, _states = self.model.predict(torch.from_numpy(obs).float())
            obs, rewards, done, trunc, info = base_env.step(ac)

            frame = base_env.render()

            seg.append(frame, rewards, obs, ac)

            # add frame to buffer
            if done or i % self.collect_seg_interval == 0:
                seg.finalise()
                self.seg_pipe.put(seg)
                seg = Segment()

                if done:
                    obs = base_env.reset()

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.num_timesteps % self.save_interval == 0:
            self.model.save(self.save_path)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def main(args):
    args = parse_args(args)

    set_random_seed(args.seed)
    Segment.set_max_len(args.seg_length)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")

    run(args, device)


def run(args, device):
    seg_pipe = Queue(maxsize=1)
    pref_pipe = Queue(maxsize=1)

    ppo_proc = Process(
        target=start_training,
        daemon=True,
        args=(args, True, seg_pipe, pref_pipe, device),
    )
    ppo_proc.start()

    pi_proc = Process(
        target=start_preference_labeling_process,
        daemon=True,
        args=(args, seg_pipe, pref_pipe),
    )
    pi_proc.start()
    ppo_proc.join()
    pi_proc.terminate()


def start_training(
    args,
    gen_segments: bool,
    seg_pipe: Queue,
    pref_pipe: Queue,
    device,
    log_dir=None,
):
    torch.set_default_dtype(torch.float32)
    Segment.set_max_len(args.seg_length)
    env = CustomEnv(args)
    policy = PPO("MlpPolicy", env, verbose=1, device=device)

    # ckpt_dir = osp.join(log_dir, "policy_checkpoints")
    # os.makedirs(ckpt_dir)
    # misc_logs_dir = osp.join(log_dir, "a2c_misc")

    n_train = args.max_prefs * (1 - args.prefs_val_fraction)
    n_val = args.max_prefs * args.prefs_val_fraction
    pref_db_train = PrefDB(maxlen=n_train)
    pref_db_val = PrefDB(maxlen=n_val)

    pref_buffer = PrefBuffer(db_train=pref_db_train, db_val=pref_db_val)
    pref_buffer.start_recv_thread(pref_pipe)

    for i in range(args.num_epochs):
        # Have policy run for x number of steps first to collect some trajectories
        # Then start the full process
        print("Train epoch number: ", i)
        if gen_segments:
            policy.learn(
                args.train_steps_per_epoch,
                callback=PretrainCallback(
                    args.num_explore_steps,
                    seg_pipe,
                    args.collect_seg_interval,
                    args.save_interval,
                ),
            )
        else:
            policy.learn(args.train_steps_per_epoch)

        # pref_buffer.stop_recv_thread()
        # pref_db_train.save("train_preferences.pkl")
        # pref_db_val.save("val_preferences.pkl")

        for i in tqdm.trange(args.num_reward_epochs_per_epoch, dynamic_ncols=True):
            env.reward_predictor.train_one_epoch(
                copy.deepcopy(pref_db_train),
                copy.deepcopy(pref_db_val),
                args.reward_model_val_interval,
            )


def start_preference_labeling_process(args, seg_pipe, pref_pipe, log_dir=None):
    torch.set_default_dtype(torch.float32)
    # Needs to be done in the main process because does GUI setup work
    # prefs_log_dir = os.path.join(log_dir, "pref_interface")
    pi = PrefInterface(max_segs=args.max_segs, use_human=True)

    # The preference interface needs to get input from stdin. stdin is
    # automatically closed at the beginning of child processes in Python,
    # so this is a bit of a hack, but it seems to be fine.
    sys.stdin = os.fdopen(0)
    pi.run(seg_pipe=seg_pipe, pref_pipe=pref_pipe)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.set_default_dtype(torch.float32)
    main(sys.argv[1:])
