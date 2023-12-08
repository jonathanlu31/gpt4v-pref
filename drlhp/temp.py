from multiprocessing import Process, Queue

# from queue import Queue
from pref_interface import PrefInterface
import sys
import os
from os import path as osp
from utils import VideoRenderer
import numpy as np
import multiprocessing


# def start_policy_training(cluster_dict, make_reward_predictor, gen_segments,
#                           start_policy_training_pipe, seg_pipe,
#                           episode_vid_queue, log_dir, a2c_params):
#     env_id = a2c_params['env_id']
#     if env_id in ['MovingDotNoFrameskip-v0', 'MovingDot-v0']:
#         policy_fn = MlpPolicy
#     elif env_id in ['PongNoFrameskip-v4', 'EnduroNoFrameskip-v4']:
#         policy_fn = CnnPolicy
#     else:
#         msg = "Unsure about policy network for {}".format(a2c_params['env_id'])
#         raise Exception(msg)

#     # configure_a2c_logger(log_dir)

#     # Done here because daemonic processes can't have children
#     env = make_envs(a2c_params['env_id'],
#                     a2c_params['n_envs'],
#                     a2c_params['seed'])
#     del a2c_params['env_id'], a2c_params['n_envs']

#     ckpt_dir = osp.join(log_dir, 'policy_checkpoints')
#     os.makedirs(ckpt_dir)

#     def f():
#         if make_reward_predictor:
#             reward_predictor = make_reward_predictor('a2c', cluster_dict)
#         else:
#             reward_predictor = None
#         misc_logs_dir = osp.join(log_dir, 'a2c_misc')
#         # easy_tf_log.set_dir(misc_logs_dir)
#         learn(
#             policy=policy_fn,
#             env=env,
#             seg_pipe=seg_pipe,
#             start_policy_training_pipe=start_policy_training_pipe,
#             episode_vid_queue=episode_vid_queue,
#             reward_predictor=reward_predictor,
#             ckpt_save_dir=ckpt_dir,
#             gen_segments=gen_segments,
#             **a2c_params)

#     proc = Process(target=f, daemon=True)
#     proc.start()
#     return env, proc

# def start_pref_interface(seg_pipe, pref_pipe, max_segs=1000,
#                          log_dir=None):
#     def f():
#         # The preference interface needs to get input from stdin. stdin is
#         # automatically closed at the beginning of child processes in Python,
#         # so this is a bit of a hack, but it seems to be fine.
#         sys.stdin = os.fdopen(0)
#         pi.run(seg_pipe=seg_pipe, pref_pipe=pref_pipe)

#     # Needs to be done in the main process because does GUI setup work
#     # prefs_log_dir = osp.join(log_dir, 'pref_interface')
#     pi = PrefInterface(max_segs=max_segs,
#                        log_dir=log_dit)
#                     #    log_dir=prefs_log_dir)
#     proc = Process(target=f, daemon=True)
#     proc.start()
#     return pi, proc

# def make_envs(env_id, n_envs, seed):
#     def wrap_make_env(env_id, rank):
#         def _thunk():
#             return make_env(env_id, seed + rank)
#         return _thunk
#     set_global_seeds(seed)
#     env = SubprocVecEnv(env_id, [wrap_make_env(env_id, i)
#                                  for i in range(n_envs)])
#     return env

# def start_reward_predictor_training(cluster_dict,
#                                     make_reward_predictor,
#                                     just_pretrain,
#                                     pref_pipe,
#                                     start_policy_training_pipe,
#                                     max_prefs,
#                                     n_initial_prefs,
#                                     n_initial_epochs,
#                                     prefs_dir,
#                                     load_ckpt_dir,
#                                     val_interval,
#                                     ckpt_interval,
#                                     log_dir):
#     def f():
#         rew_pred = make_reward_predictor('train', cluster_dict)
#         rew_pred.init_network(load_ckpt_dir)

#         if prefs_dir is not None:
#             train_path = osp.join(prefs_dir, 'train.pkl.gz')
#             pref_db_train = PrefDB.load(train_path)
#             print("Loaded training preferences from '{}'".format(train_path))
#             n_prefs, n_segs = len(pref_db_train), len(pref_db_train.segments)
#             print("({} preferences, {} segments)".format(n_prefs, n_segs))

#             val_path = osp.join(prefs_dir, 'val.pkl.gz')
#             pref_db_val = PrefDB.load(val_path)
#             print("Loaded validation preferences from '{}'".format(val_path))
#             n_prefs, n_segs = len(pref_db_val), len(pref_db_val.segments)
#             print("({} preferences, {} segments)".format(n_prefs, n_segs))
#         else:
#             n_train = max_prefs * (1 - PREFS_VAL_FRACTION)
#             n_val = max_prefs * PREFS_VAL_FRACTION
#             pref_db_train = PrefDB(maxlen=n_train)
#             pref_db_val = PrefDB(maxlen=n_val)

#         pref_buffer = PrefBuffer(db_train=pref_db_train,
#                                  db_val=pref_db_val)
#         pref_buffer.start_recv_thread(pref_pipe)
#         if prefs_dir is None:
#             pref_buffer.wait_until_len(n_initial_prefs)

#         save_prefs(log_dir, pref_db_train, pref_db_val)

#         if not load_ckpt_dir:
#             print("Pretraining reward predictor for {} epochs".format(
#                 n_initial_epochs))
#             pref_db_train, pref_db_val = pref_buffer.get_dbs()
#             for i in range(n_initial_epochs):
#                 # Note that we deliberately don't update the preferences
#                 # databases during pretraining to keep the number of
#                 # fairly preferences small so that pretraining doesn't take too
#                 # long.
#                 print("Reward predictor training epoch {}".format(i))
#                 rew_pred.train(pref_db_train, pref_db_val, val_interval)
#                 if i and i % ckpt_interval == 0:
#                     rew_pred.save()
#             print("Reward predictor pretraining done")
#             rew_pred.save()

#         if just_pretrain:
#             return

#         start_policy_training_pipe.put(True)

#         i = 0
#         while True:
#             pref_db_train, pref_db_val = pref_buffer.get_dbs()
#             save_prefs(log_dir, pref_db_train, pref_db_val)
#             rew_pred.train(pref_db_train, pref_db_val, val_interval)
#             if i and i % ckpt_interval == 0:
#                 rew_pred.save()

#     proc = Process(target=f, daemon=True)
#     proc.start()
#     return proc


def start_episode_renderer():
    episode_vid_queue = Queue()
    renderer = VideoRenderer(
        episode_vid_queue,
        playback_speed=2,
        zoom=2,
        mode=VideoRenderer.play_through_mode,
    )
    return episode_vid_queue, renderer


import pygame

print("before error")
import moviepy.editor

print("after error")
import cv2


import skvideo


skvideo.setFFmpegPath(
    "/home/jonathan/285-project/drlhp/ffmpeg/ffmpeg-git-20231006-amd64-static/"
)

print("FFmpeg path: {}".format(skvideo.getFFmpegPath()))
print("FFmpeg version: {}".format(skvideo.getFFmpegVersion()))

import skvideo.io


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")

    # seg_pipe = Queue(maxsize=1)
    # pref_pipe = Queue(maxsize=1)

    # episode_vid_queue, episode_renderer = start_episode_renderer()

    # videodata = skvideo.io.vread("vid.mp4")[:, :, :, -1]
    # print(skvideo.io.vread("vid.mp4").shape)
    videodata = skvideo.io.vread("vid.mp4")

    # videodata = np.load("bw.dat")

    # episode_vid_queue.put(videodata[:, :, :, -1])
    # episode_renderer.render()

    # size = (videodata.shape[1], videodata.shape[0])
    size = (videodata.shape[1], videodata.shape[2])
    duration = 5
    fps = videodata.shape[0] // duration
    out = cv2.VideoWriter(
        "out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[1], size[0]), True
    )
    # for _ in range(fps * duration):
    #     data = np.random.randint(0, 256, size, dtype='uint8')
    for f in range(videodata.shape[0]):
        out.write(videodata[f])
    # out.write(data)
    out.release()
    print("done")

    print(videodata)

    pygame.init()
    video = moviepy.editor.VideoFileClip("out.mp4")
    video.preview()
    pygame.quit()

    # episode_renderer.stop()

    # cluster_dict = create_cluster_dict(['ps', 'a2c', 'train'])
    # ps_proc = start_parameter_server(cluster_dict, make_reward_predictor)
    # env, a2c_proc = start_policy_training(
    #     cluster_dict=cluster_dict,
    #     make_reward_predictor=make_reward_predictor,
    #     gen_segments=True,
    #     start_policy_training_pipe=start_policy_training_flag,
    #     seg_pipe=seg_pipe,
    #     episode_vid_queue=episode_vid_queue,
    #     log_dir=general_params['log_dir'],
    #     a2c_params=a2c_params)

    # pi, pi_proc = start_pref_interface(
    #         seg_pipe=seg_pipe,
    #         pref_pipe=pref_pipe)

    # rpt_proc = start_reward_predictor_training(
    #         cluster_dict=cluster_dict,
    #         make_reward_predictor=make_reward_predictor,
    #         just_pretrain=False,
    #         pref_pipe=pref_pipe,
    #         start_policy_training_pipe=start_policy_training_flag,
    #         max_prefs=general_params['max_prefs'],
    #         prefs_dir=general_params['prefs_dir'],
    #         load_ckpt_dir=rew_pred_training_params['load_ckpt_dir'],
    #         n_initial_prefs=general_params['n_initial_prefs'],
    #         n_initial_epochs=rew_pred_training_params['n_initial_epochs'],
    #         val_interval=rew_pred_training_params['val_interval'],
    #         ckpt_interval=rew_pred_training_params['ckpt_interval'],
    #         log_dir=general_params['log_dir'])
