"""
Author: Xu Yan
"""

import cv2
import numpy as np
import time
import random
import subprocess

class VideoRecorder():
    def __init__(self, video_name="rgb_cam", framerate=10, frame_height=1408, frame_width=1408):
        
        self.video_file1 = video_name + "_with_gaze.mp4"
        self.video_file2 = video_name + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer1 = cv2.VideoWriter(self.video_file1, fourcc, framerate, (frame_width, frame_height))
        self.writer2 = cv2.VideoWriter(self.video_file2, fourcc, framerate, (frame_width, frame_height))

    def record_frame(self, frame, frame_no_gaze):
        self.writer1.write(frame)
        self.writer2.write(frame_no_gaze)

    def end_recording(self):
        self.writer1.release()
        self.writer2.release()
        print(f'Recording Terminated. Videos saved to {self.video_file1} and {self.video_file2}')



class GazeRecorder():
    def __init__(self, gaze_name="raw_gaze"):
# TODO: add open file
        self.gaze_file = gaze_name + ".npy"

        self.gazes = []
        self.count = 0

    def record_frame(self, gaze):
        if gaze is None:
            gaze = np.array([np.nan, np.nan], dtype=float)
        self.gazes.append((self.count, gaze))
        self.count += 1

    def end_recording(self):
        np.save(self.gaze_file, np.array(self.gazes, dtype=object))
        print(f'Gaze coordinates saved to {self.gaze_file}')