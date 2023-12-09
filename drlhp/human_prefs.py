import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import moviepy.editor
import cv2
import numpy as np
import skvideo.io
from PIL import Image


class VideoRenderer:
    DURATION = 5
    TMP = "tmp.mp4"
    SUPPORTED_FILE_TYPES = {"mp4", "gif"}
    WIDTH = 800
    HEIGHT = 600

    @staticmethod
    def render_np_array(arr: np.ndarray, is_img=False):
        size = (arr.shape[1], arr.shape[2])
        fps = arr.shape[0] // VideoRenderer.DURATION
        out = cv2.VideoWriter(
            VideoRenderer.TMP,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (size[1], size[0]),
            True,
        )
        red = arr[:, :, :, 2].copy()
        blue = arr[:, :, :, 0].copy()
        arr[:, :, :, 0] = red
        arr[:, :, :, 2] = blue
        for frame in range(arr.shape[0]):
            out.write(arr[frame])
        out.release()
        pref = VideoRenderer.cv2_loop(VideoRenderer.TMP)
        os.remove(VideoRenderer.TMP)
        return pref

    @staticmethod
    def render_mp4(filename: str):
        pygame.init()
        video = moviepy.editor.VideoFileClip(VideoRenderer.TMP)
        video.preview()
        pygame.quit()

    @staticmethod
    def loop_mp4(filename: str, loop=3):
        pygame.init()
        video = moviepy.editor.VideoFileClip(VideoRenderer.TMP)
        video = video.loop(loop)
        video.preview()
        pygame.quit()

    @staticmethod
    def cv2_loop(filename: str):
        pygame.init()
        clicked = False
        while not clicked:
            video = cv2.VideoCapture(filename)
            # w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            # h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)
            screen = pygame.display.set_mode(
                (VideoRenderer.WIDTH, VideoRenderer.HEIGHT)
            )
            clock = pygame.time.Clock()
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            pygame.quit()
                            return -1
                        elif event.key == pygame.K_RIGHT:
                            pygame.quit()
                            return 1
                        elif event.key == pygame.K_RETURN:
                            pygame.quit()
                            return 0

                ret, frame = video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(
                        frame, (VideoRenderer.WIDTH, VideoRenderer.HEIGHT)
                    )
                    image = pygame.image.frombuffer(
                        frame, (VideoRenderer.WIDTH, VideoRenderer.HEIGHT), "RGB"
                    )
                    screen.blit(image, (0, 0))
                    clock.tick(fps)
                    pygame.display.flip()
                else:
                    running = False
        pygame.quit()

    @staticmethod
    def render_two_np_array(arr1: np.ndarray, arr2: np.ndarray):
        new_height = max(arr1.shape[1], arr2.shape[1])
        new_frames = max(arr1.shape[0], arr2.shape[0])
        joint = np.zeros(
            (new_frames, new_height, arr1.shape[2] + arr2.shape[2], arr1.shape[3])
        )
        padded_arr1 = np.pad(
            arr1,
            (
                (0, new_frames - arr1.shape[0]),
                (0, new_height - arr1.shape[1]),
                (0, 0),
                (0, 0),
            ),
            constant_values=0,
        )
        padded_arr2 = np.pad(
            arr2,
            (
                (0, new_frames - arr2.shape[0]),
                (0, new_height - arr2.shape[1]),
                (0, 0),
                (0, 0),
            ),
            constant_values=(0, 0),
        )

        for frame in range(new_frames):
            joint[frame] = np.append(padded_arr1[frame], padded_arr2[frame], axis=1)

        del padded_arr1
        del padded_arr2
        return VideoRenderer.render_np_array(np.uint8(joint))

    @staticmethod
    def mp4_to_np(filename: str):
        assert (
            filename.split(".")[-1] in VideoRenderer.SUPPORTED_FILE_TYPES
        ), f"{filename.split('.')[-1]} not supported"
        return skvideo.io.vread(filename)


def get_prefs(arr1: np.ndarray, arr2: np.ndarray):
    return VideoRenderer.render_two_np_array(arr1, arr2)


class ImageRenderer:
    @staticmethod
    def preprocess(img: np.ndarray, frames=20):
        unsqueezed = np.expand_dims(img[:, :, 0:3], axis=0)
        out = unsqueezed
        for _ in range(frames - 1):
            out = np.append(out, unsqueezed, axis=0)
        return out

    @staticmethod
    def file_to_np(filename: str):
        return ImageRenderer.preprocess(np.asarray(Image.open(filename)))


def vid_ex():
    return np.load("bowling.npz")["main"], VideoRenderer.mp4_to_np("taco-tues.gif")


def img_ex():
    return ImageRenderer.file_to_np("vlm.png"), ImageRenderer.file_to_np(
        "Brahms_Lutoslawski_Final.png"
    )


if __name__ == "__main__":
    while True:
        if np.random.rand() < 0.5:
            videodata1, videodata2 = vid_ex()
        else:
            videodata1, videodata2 = img_ex()
        if np.random.rand() < 0.5:
            print(get_prefs(videodata1, videodata2))
        else:
            print(get_prefs(videodata2, videodata1))
