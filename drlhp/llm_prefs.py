from openai import OpenAI
import base64
import requests
import cv2
import os
from dotenv import load_dotenv
from human_prefs import VideoRenderer, ImageRenderer
import numpy as np
from typing import Literal

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

PREFERENCE_PROMPT = """You are an expert in the English language. I will give you $100 if you get this correct. This is an image of two lines on a checkboard background. Which of the two lines looks more like an A. They both probably won't look much like the letter but give your best judgement. I believe in you. Output a single answer: LEFT or RIGHT."""


class GPT:
    KEY = os.environ.get("KEY")
    CLIENT = OpenAI(api_key=KEY)
    HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
    URL = "https://api.openai.com/v1/chat/completions"
    MODEL = "gpt-4-vision-preview"
    CHOICES = {"FIRST", "SECOND", "NEITHER"}

    @staticmethod
    def _encode_img(path: str):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def query_img_preferences(
        path1: str,
        path2: str = None,
        query=PREFERENCE_PROMPT,
        bytes1: str = None,
        bytes2: str = None,
    ):
        msg_content = [
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{bytes1 or GPT._encode_img(path1)}",
                    "detail": "low",
                },
            },
        ]
        if path2 or bytes2:
            msg_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{bytes2 or GPT._encode_img(path2)}",
                        "detail": "low",
                    },
                }
            )
        response = GPT.CLIENT.chat.completions.create(
            model=GPT.MODEL,
            messages=[{"role": "user", "content": msg_content}],
            max_tokens=300,
        )
        pref = response.choices[0].message.content
        print(pref)
        print(response.usage.total_tokens)
        # TODO: log the full content as well for debugging if we decide to use step-by-step reasoning
        return GPT.pref_to_int(pref)

    @staticmethod
    def pref_to_int(pref: str) -> Literal[-1, 1, 0, None]:
        # assert pref in GPT.CHOICES, f'LLM responded with "{pref}", which is not a valid response'
        if "LEFT" in pref or "FIRST" in pref:
            return -1
        elif "RIGHT" in pref or "SECOND" in pref:
            return 1
        elif "NEITHER" in pref:
            return 0
        return None

    @staticmethod
    def query_videos(
        path1: str,
        path2: str,
        query="What are in these videos? Is there any difference between them?",
    ):
        query = "Which video is more related to Berkeley? Please respond with a single word. Here are your three choices: FIRST, SECOND, or NEITHER"
        frames1, frames2 = GPT.vid_to_frames(path1), GPT.vid_to_frames(path2)
        frame_to_payload = lambda x: {"image": x, "resize": 768}
        messages = [
            {
                "role": "user",
                "content": [
                    f"These are the frames for two videos that I want to upload.",
                    "FIRST",
                    *map(frame_to_payload, frames1[0::25]),
                    "SECOND",
                    *map(frame_to_payload, frames2[0::25]),
                    query,
                ],
            }
        ]
        params = {"model": GPT.MODEL, "messages": messages, "max_tokens": 30}  # 200
        pref = GPT.CLIENT.chat.completions.create(**params).choices[0].message.content
        return GPT.pref_to_int(pref)

    @staticmethod
    def vid_to_frames(path: str):
        vid = cv2.VideoCapture(path)
        b64frames = []
        while vid.isOpened():
            success, frame = vid.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            b64frames.append(base64.b64encode(buffer).decode("utf-8"))
        vid.release()
        return b64frames

    @staticmethod
    def combine_and_query(arr1: np.ndarray, arr2: np.ndarray, query=PREFERENCE_PROMPT):
        combined = VideoRenderer.combine_two_np_array(arr1, arr2)
        cv2_arr = VideoRenderer.convert_np_to_cv2(combined)
        cv2.imwrite(VideoRenderer.TMP_PNG, cv2_arr[0])
        success, buffer = cv2.imencode(".png", cv2_arr[0])
        img_bytes = base64.b64encode(buffer).decode("utf-8")
        if query:
            pref = GPT.query_img_preferences(
                VideoRenderer.TMP_PNG, query=query, bytes1=img_bytes
            )
        else:
            pref = GPT.query_img_preferences(VideoRenderer.TMP_PNG, bytes1=img_bytes)
        # os.remove(VideoRenderer.TMP_PNG)
        return pref


if __name__ == "__main__":
    path1 = "test.png"
    path2 = "test copy.png"

    arr1, arr2 = ImageRenderer.file_to_np(path1), ImageRenderer.file_to_np(path2)

    print(
        GPT.combine_and_query(
            arr1,
            arr2,
            query=PREFERENCE_PROMPT,
        )
    )

    # print(
    #     GPT.query_img_preferences(
    #         "test.png",
    #         query=PREFERENCE_PROMPT,
    #     )
    # )
    # print(GPT.query_images('Brahms_Lutoslawski_Final.png', 'vlm.png'))
    # print(GPT.query_images('drlhp/test.png', 'drlhp/test copy.png'))
    # print(GPT.query_videos('vid.mp4', 'taco-tues.gif'))
