import collections
import copy
import gzip
import pickle
import queue
import time
import zlib
from threading import Lock, Thread
from multiprocessing import Queue
from torch.utils.data import Dataset
from collections import deque
import torch

# import easy_tf_log
import numpy as np


class Segment:
    """
    A short recording of agent's behaviour in the environment,
    consisting of a number of video frames and the rewards it received
    during those frames.
    """

    max_len = None

    @classmethod
    def set_max_len(cls, max_len):
        cls.max_len = max_len

    def __init__(self):
        assert Segment.max_len is not None, "Set segment length first"
        self.frames = deque()
        self.rewards = deque()
        self.observations = deque()
        self.actions = deque()
        self.hash = None

    def append(self, frame, reward, ob, act):
        self.frames.append(frame)
        self.rewards.append(reward)
        self.observations.append(ob)
        self.actions.append(act)

        if len(self.rewards) > Segment.max_len:
            self.frames.popleft()
            self.rewards.popleft()
            self.observations.popleft()
            self.actions.popleft()

    def finalise(self, seg_id=None):
        if seg_id is not None:
            self.hash = seg_id
        else:
            # This looks expensive, but don't worry -
            # it only takes about 0.5 ms.
            self.hash = hash(np.array(self.frames).tostring())

    def get_trajectory_format(self):
        """Interleave the observations and actions to be in the format expected by the reward predictors

        i.e. [num_timesteps x (ob_dim + ac_dim)]
        """
        ob_acts = zip(self.observations, self.actions)
        ob_acts_np = np.array([np.hstack([ob, ac]) for ob, ac in ob_acts])
        return ob_acts_np

    def __len__(self):
        return len(self.frames)


class CompressedDict(collections.abc.MutableMapping):
    def __init__(self):
        self.store: dict[str, Segment] = dict()

    def __getitem__(self, key) -> Segment:
        return pickle.loads(zlib.decompress(self.store[key]))

    def __setitem__(self, key, value):
        self.store[key] = zlib.compress(pickle.dumps(value))

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self) -> int:
        return len(self.store)

    def __keytransform__(self, key):
        return key


class PrefDB(Dataset):
    """
    A circular database of preferences about pairs of segments.

    For each preference, we store the preference itself
    (mu in the paper) and the two segments the preference refers to.
    Segments are stored with deduplication - so that if multiple
    preferences refer to the same segment, the segment is only stored once.
    """

    def __init__(self, maxlen):
        super().__init__()
        self.segments = CompressedDict()
        self.seg_refs = {}
        self.prefs: list[tuple[str, str, int]] = []
        self.maxlen = maxlen

    def append(self, s1: Segment, s2: Segment, pref):
        k1 = hash(np.array(s1).tostring())
        k2 = hash(np.array(s2).tostring())

        for k, s in zip([k1, k2], [s1, s2]):
            if k not in self.segments.keys():
                self.segments[k] = s
                self.seg_refs[k] = 1
            else:
                self.seg_refs[k] += 1

        tup = (k1, k2, pref)
        self.prefs.append(tup)

        if len(self.prefs) > self.maxlen:
            self.del_first()

    def del_first(self):
        self.del_pref(0)

    def del_pref(self, n):
        if n >= len(self.prefs):
            raise IndexError("Preference {} doesn't exist".format(n))
        k1, k2, _ = self.prefs[n]
        for k in [k1, k2]:
            if self.seg_refs[k] == 1:
                del self.segments[k]
                del self.seg_refs[k]
            else:
                self.seg_refs[k] -= 1
        del self.prefs[n]

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, idx: int):
        k1, k2, pref = self.prefs[idx]
        s1, s2 = self.segments[k1], self.segments[k2]
        s1_pt, s2_pt = torch.from_numpy(s1.get_trajectory_format()), torch.from_numpy(
            s2.get_trajectory_format()
        )

        if pref == 1:
            pref_pt = torch.Tensor([0, 1])
        elif pref == 0:
            pref_pt = torch.Tensor([0.5, 0.5])
        else:
            pref_pt = torch.Tensor([1, 0])
        return s1_pt, s2_pt, pref_pt

    def save(self, path):
        copy = copy.deepcopy(self)
        with gzip.open(path, "wb") as pkl_file:
            pickle.dump(copy, pkl_file)

    @staticmethod
    def load(path):
        with gzip.open(path, "rb") as pkl_file:
            pref_db = pickle.load(pkl_file)
        return pref_db


class PrefBuffer:
    """
    A helper class to manage asynchronous receiving of preferences on a
    background thread.
    """

    def __init__(self, db_train, db_val):
        self.train_db = db_train
        self.val_db = db_val
        self.lock = Lock()
        self.stop_recv = False

    def start_recv_thread(self, pref_pipe):
        self.stop_recv = False
        Thread(target=self.recv_prefs, args=(pref_pipe,)).start()

    def stop_recv_thread(self):
        self.stop_recv = True

    def recv_prefs(self, pref_pipe: Queue):
        n_recvd = 0
        while not self.stop_recv:
            try:
                s1, s2, pref = pref_pipe.get(block=True, timeout=1)
            except queue.Empty:
                continue
            n_recvd += 1

            val_fraction = self.val_db.maxlen / (
                self.val_db.maxlen + self.train_db.maxlen
            )

            self.lock.acquire(blocking=True)
            if np.random.rand() < val_fraction:
                self.val_db.append(s1, s2, pref)
                # easy_tf_log.tflog('val_db_len', len(self.val_db))
            else:
                self.train_db.append(s1, s2, pref)
                # easy_tf_log.tflog('train_db_len', len(self.train_db))
            self.lock.release()

            # easy_tf_log.tflog('n_prefs_recvd', n_recvd)

    def train_db_len(self):
        return len(self.train_db)

    def val_db_len(self):
        return len(self.val_db)

    def get_dbs(self):
        self.lock.acquire(blocking=True)
        train_copy = copy.deepcopy(self.train_db)
        val_copy = copy.deepcopy(self.val_db)
        self.lock.release()
        return train_copy, val_copy

    def wait_until_len(self, min_len: int):
        while True:
            self.lock.acquire()
            train_len = len(self.train_db)
            val_len = len(self.val_db)
            self.lock.release()
            if train_len >= min_len and val_len != 0:
                break
            print("Waiting for preferences; {} so far".format(train_len))
            time.sleep(5.0)
