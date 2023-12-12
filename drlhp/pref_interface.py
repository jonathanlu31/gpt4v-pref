#!/usr/bin/env python

"""
A simple CLI-based interface for querying the user about segment preferences.
"""

import logging
import queue
import time
from itertools import combinations
from multiprocessing import Queue
from random import shuffle

# import easy_tf_log
import numpy as np

from human_prefs import get_prefs, VideoRenderer
from pref_db import Segment
from llm_prefs import GPT, PREFERENCE_PROMPT


class PrefInterface:
    def __init__(self, max_segs: int, use_human: bool = False, log_dir=None):
        self.seg_idx = 0
        self.use_human = use_human
        self.segments: list[Segment] = []
        self.tested_pairs = set()  # For O(1) lookup
        self.max_segs = max_segs
        # easy_tf_log.set_dir(log_dir)

    def run(self, seg_pipe, pref_pipe):
        while len(self.segments) < 2:
            print("Preference interface waiting for segments")
            time.sleep(5.0)
            self.recv_segments(seg_pipe)

        while True:
            seg_pair = None
            while seg_pair is None:
                try:
                    seg_pair = self.sample_seg_pair()
                except IndexError:
                    print(
                        "Preference interface ran out of untested segments;"
                        "waiting..."
                    )
                    # If we've tested all possible pairs of segments so far,
                    # we'll have to wait for more segments
                    time.sleep(1.0)
                    self.recv_segments(seg_pipe)
            s1, s2 = seg_pair

            logging.debug(
                "Querying preference for segments %s and %s", s1.hash, s2.hash
            )

            if self.use_human:
                pref = get_prefs(np.array(s1.frames), np.array(s2.frames), is_img=True)
            else:
                pref = GPT.combine_and_query(np.array(s1.frames), np.array(s2.frames))

            if pref is not None:
                pref_pipe.put((s1, s2, pref))
            # If pref is None, the user answered "incomparable" for the segment
            # pair. The pair has been marked as tested; we just drop it.

            self.recv_segments(seg_pipe)

    def recv_segments(self, seg_pipe):
        """
        Receive segments from `seg_pipe` into circular buffer `segments`.
        """
        max_wait_seconds = 0.5
        start_time = time.time()
        n_recvd = 0
        while time.time() - start_time < max_wait_seconds:
            try:
                segment = seg_pipe.get(block=True, timeout=max_wait_seconds)
            except queue.Empty:
                return
            if len(self.segments) < self.max_segs:
                self.segments.append(segment)
            else:
                self.segments[self.seg_idx] = segment
                self.seg_idx = (self.seg_idx + 1) % self.max_segs
            n_recvd += 1

    # easy_tf_log.tflog('segment_idx', self.seg_idx)
    # easy_tf_log.tflog('n_segments_rcvd', n_recvd)
    # easy_tf_log.tflog('n_segments', len(self.segments))

    def sample_seg_pair(self) -> tuple[Segment, Segment]:
        """
        Sample a random pair of segments which hasn't yet been tested.
        """
        segment_idxs = list(range(len(self.segments)))
        shuffle(segment_idxs)
        possible_pairs = combinations(segment_idxs, 2)
        for i1, i2 in possible_pairs:
            s1, s2 = self.segments[i1], self.segments[i2]
            if ((s1.hash, s2.hash) not in self.tested_pairs) and (
                (s2.hash, s1.hash) not in self.tested_pairs
            ):
                self.tested_pairs.add((s1.hash, s2.hash))
                self.tested_pairs.add((s2.hash, s1.hash))
                return s1, s2
        raise IndexError("No segment pairs yet untested")
