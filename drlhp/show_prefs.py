#!/usr/bin/env python3
"""
Display examples of the specified preference database
(with the less-preferred segment on the left,
and the more-preferred segment on the right)
(skipping over equally-preferred segments)
"""

import argparse

import numpy as np

from human_prefs import VideoRenderer
from pref_db import PrefDB
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefs", help=".pkl.gz file")
    args = parser.parse_args()

    prefs = PrefDB.load(args.prefs)

    print("{} preferences found".format(len(prefs)))
    print("Preferred segment on the right")

    for k1, k2, pref in prefs.prefs:
        s1, s2 = prefs.segments[k1], prefs.segments[k2]
        if pref == (0.5, 0.5):
            continue

        if pref == (0.0, 1.0):
            img = VideoRenderer.combine_two_np_array(
                np.array(s1.frames), np.array(s2.frames)
            )
        elif pref == (1.0, 0.0):
            img = VideoRenderer.combine_two_np_array(
                np.array(s2.frames), np.array(s1.frames)
            )
        else:
            raise Exception("Unexpected preference", pref)

        cv2_img = VideoRenderer.convert_np_to_cv2(img)
        cv2.imshow("Preference (right is right)", cv2_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        input()


if __name__ == "__main__":
    main()
