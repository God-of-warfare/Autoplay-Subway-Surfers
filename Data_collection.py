import cv2
import numpy as np
import mss
import time
import keyboard
import os
from collections import deque
from datetime import datetime


def create_directory_structure():
    main_dirs = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'NOTHING']
    for dir_name in main_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def save_frames(frames, action):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = os.path.join(action, f"capture_{timestamp}")
    os.makedirs(folder_name, exist_ok=True)

    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(folder_name, f'frame_{i + 1}.png'), frame)


def capture_frames():
    create_directory_structure()

    monitor = {"top": 3, "left": 0, "width": 584, "height": 1080}
    fps = 75
    frame_interval = 10  # Number of frames between each action
    frames_to_capture = 3  # Number of frames to capture per action

    sct = mss.mss()
    frame_buffer = deque(maxlen=frames_to_capture * frame_interval * 3)

    nothing_capture_timeout = 1
    last_action_time = time.time()

    print("Start playing the game. Press 'a', 's', 'd', 'w' to capture frames or wait for 'NOTHING'...")

    while True:
        start_time = time.time()

        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        frame_buffer.append(frame)

        action = None
        if keyboard.is_pressed('a'):
            action = 'LEFT'
        elif keyboard.is_pressed('s'):
            action = 'DOWN'
        elif keyboard.is_pressed('d'):
            action = 'RIGHT'
        elif keyboard.is_pressed('w'):
            action = 'UP'

        if action:
            print(f"Key pressed: {action}. Saving frames...")
            # Include the current frame (key press frame) and previous frames
            selected_frames = list(frame_buffer)[-frames_to_capture * frame_interval - 1::frame_interval]
            selected_frames = selected_frames[-frames_to_capture:]  # Ensure we have exactly frames_to_capture
            save_frames(selected_frames, action)
            last_action_time = time.time()
        elif time.time() - last_action_time >= nothing_capture_timeout:
            print("No key pressed within timeout. Saving 'NOTHING' frames...")
            selected_frames = list(frame_buffer)[-frames_to_capture * frame_interval::frame_interval]
            save_frames(selected_frames, 'NOTHING')
            last_action_time = time.time()

        elapsed_time = time.time() - start_time
        time_to_wait = max(0, (1 / fps) - elapsed_time)
        time.sleep(time_to_wait)


if __name__ == "__main__":
    capture_frames()
