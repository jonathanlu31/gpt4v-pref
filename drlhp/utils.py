import numpy as np
import pyglet
from multiprocessing import Process
import queue
import time
import random
import socket
from scipy.ndimage import zoom


# Based on SimpleImageViewer in OpenAI gym
class Im(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display
            )
            self.width = width
            self.height = height
            self.isopen = True

        assert arr.shape == (
            self.height,
            self.width,
        ), "You passed in an image with the wrong number shape"

        image = pyglet.image.ImageData(
            self.width, self.height, "L", arr.tobytes(), pitch=-self.width
        )
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


class VideoRenderer:
    play_through_mode = 0
    restart_on_get_mode = 1

    def __init__(self, vid_queue, mode, zoom=1, playback_speed=1):
        assert (
            mode == VideoRenderer.restart_on_get_mode
            or mode == VideoRenderer.play_through_mode
        )
        self.mode = mode
        self.vid_queue = vid_queue
        self.zoom_factor = zoom
        self.playback_speed = playback_speed
        self.proc = Process(target=self.render)
        self.proc.start()

    def stop(self):
        self.proc.terminate()

    def render(self):
        v = Im()
        frames = self.vid_queue.get(block=True)
        t = 0
        while True:
            # Add a grey dot on the last line showing position
            width = frames[t].shape[1]
            fraction_played = t / len(frames)
            x = int(fraction_played * width)
            frames[t][-1][x] = 128

            zoomed_frame = zoom(frames[t], self.zoom_factor)
            v.imshow(zoomed_frame)

            if self.mode == VideoRenderer.play_through_mode:
                # Wait until having finished playing the current
                # set of frames. Then, stop, and get the most
                # recent set of frames.
                t += self.playback_speed
                if t >= len(frames):
                    frames = self.get_queue_most_recent()
                    t = 0
                else:
                    time.sleep(1 / 60)
            elif self.mode == VideoRenderer.restart_on_get_mode:
                # Always try and get a new set of frames to show.
                # If there is a new set of frames on the queue,
                # restart playback with those frames immediately.
                # Otherwise, just keep looping with the current frames.
                try:
                    frames = self.vid_queue.get(block=False)
                    t = 0
                except queue.Empty:
                    t = (t + self.playback_speed) % len(frames)
                    time.sleep(1 / 60)

    def get_queue_most_recent(self):
        # Make sure we at least get something
        item = self.vid_queue.get(block=True)
        while True:
            try:
                item = self.vid_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                break
        return item


def get_port_range(start_port, n_ports, random_stagger=False):
    # If multiple runs try and call this function at the same time,
    # the function could return the same port range.
    # To guard against this, automatically offset the port range.
    if random_stagger:
        start_port += random.randint(0, 20) * n_ports

    free_range_found = False
    while not free_range_found:
        ports = []
        for port_n in range(n_ports):
            port = start_port + port_n
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))
                ports.append(port)
            except socket.error as e:
                if e.errno == 98 or e.errno == 48:
                    print("Warning: port {} already in use".format(port))
                    break
                else:
                    raise e
            finally:
                s.close()
        if len(ports) < n_ports:
            # The last port we tried was in use
            # Try again, starting from the next port
            start_port = port + 1
        else:
            free_range_found = True

    return ports
