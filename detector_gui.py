from os.path import isfile
from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
import utils
from typing import Dict

UP_KEY = 2490368
DOWN_KEY = 2621440
RIGHT_KEY = 2555904
LEFT_KEY = 2424832
ESC_KEY = 27
Q_KEY = ord('q')


class DetectorGui:
    """Display capture frames, stepping with CAP_IMAGES and seeking only on local sources."""

    def __init__(self, source=0, capture_api=cv2.CAP_ANY, *, skip_step: int = 0):
        self.__capture = cv2.VideoCapture(source, capture_api)
        self.capture_api = capture_api
        if not self.__capture.isOpened():
            raise RuntimeError(f"Failed to open video source {source}")
        self.__can_seek = isinstance(source, str) and (capture_api == cv2.CAP_IMAGES or isfile(source))
        self.__MAX_SKIP_STEP = 100
        self.__skip_step = max(0, min(self.__MAX_SKIP_STEP, skip_step))
        self.__frame_idx = 0  # Index of the next frame to read.

    @property
    def fps(self) -> float:
        return self.__capture.get(cv2.CAP_PROP_FPS)

    @property
    def frame_index(self) -> int:
        """Return the zero-based source frame index during the frame callback."""
        return self.__frame_idx - 1

    def __update_and_clamp_skip_step(self, step):
        self.__skip_step += step
        self.__skip_step = max(1, min(self.__MAX_SKIP_STEP, self.__skip_step))

    def __skip_ahead(self):
        if self.__skip_step > 0:
            self.__seek_frame(self.__frame_idx + self.__skip_step)

    def __jump_back(self):
        self.__seek_frame(max(0, self.__frame_idx - 1 - self.__skip_step))

    def __seek_frame(self, target: int) -> None:
        if not self.__can_seek:
            print("Seeking is only enabled for local video files and image sequences.")
            return
        if not self.__capture.set(cv2.CAP_PROP_POS_FRAMES, target):
            print(f"Could not seek to frame {target}.")
            return

        position = self.__capture.get(cv2.CAP_PROP_POS_FRAMES)
        if not position.is_integer() or position < 0:
            print(f"Could not verify the position after seeking to frame {target}.")
            self.__frame_idx = target
            return
        self.__frame_idx = int(position)

    def __wait_for_key(self):
        # Step through manually when opening images from directory, otherwise stream continuously.
        delay = 0 if self.capture_api == cv2.CAP_IMAGES else 1
        return cv2.waitKeyEx(delay)

    def __draw_skip_step(self, frame: np.ndarray) -> None:
        text = f"Skip step: {self.__skip_step}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        height, width = frame.shape[:2]
        padding = min(10, width // 8, height // 8)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, 1.0, 3)
        scale = min(0.6, (width - 2 * padding) / text_width, (height - 2 * padding) / (text_height + baseline))
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, 3)
        origin = (max(0, width - padding - text_width), min(height - 1, padding + text_height))
        cv2.putText(frame, text, origin, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, origin, font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    def __key_handler(self, key):
        if key in (UP_KEY, 65362, 16777235):
            self.__update_and_clamp_skip_step(1)
        elif key in (DOWN_KEY, 65364, 16777237):
            self.__update_and_clamp_skip_step(-1)
        elif key in (RIGHT_KEY, 65363, 16777236):
            self.__skip_ahead()
        elif key in (LEFT_KEY, 65361, 16777234):
            self.__jump_back()
        elif key == ESC_KEY or key == Q_KEY:
            self.__gui_running = False

    def run(self, frame_callback: Callable[[NDArray[np.uint8]], None]) -> None:
        """Run playback; the callback may annotate the frame in place."""
        self.__gui_running = True
        window_name = "Video"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            while self.__gui_running:
                ret, frame = self.__capture.read()
                if not ret:
                    print("Reached end of video.")
                    break
                self.__frame_idx += 1

                if frame is None or not utils.is_valid_image(frame):
                    print("Invalid frame read, skipping...")
                    continue

                frame_callback(frame)

                display_frame = frame.copy()
                self.__draw_skip_step(display_frame)
                cv2.imshow(window_name, display_frame)
                key = self.__wait_for_key()
                self.__key_handler(key)
                if self.__gui_running and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.__capture.release()
            cv2.destroyAllWindows()
