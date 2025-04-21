"""
Author: Jonathan Ouyang
Modified by: Xu Yan
"""

import cv2
import ffmpeg
import numpy as np
import time
import random
import subprocess

class DataRecorder():
    def __init__(self, video_name="rgb_cam", gaze_name="gaze_data", framerate=10, cam_type="aria"):
        
        self.video_file1 = video_name + ".mp4"
        self.video_file2 = video_name + "_no_gaze.mp4"
        self.gaze_file = gaze_name + ".npy"
        self.framerate = int(framerate)

        self.gazes = []

        # Get frame dimensions from the first image. Assumes all are same size
        if cam_type == "aria":
            self.frame_height, self.frame_width, _ = 1408, 1408, 3
        if cam_type == "realsense":
            self.frame_height, self.frame_width, _ = 480, 640, 3

        # Setup the ffmpeg pipe for video writing
        self.process1 = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(self.frame_width, self.frame_height), framerate = self.framerate)
            .output(self.video_file1, vcodec='libx264', pix_fmt='yuv420p') # Using h264 codec for mp4
            .global_args('-nostats', '-loglevel', 'error')
            .overwrite_output()
            .run_async(pipe_stdin=True,
                       pipe_stdout=subprocess.DEVNULL,
                       pipe_stderr=subprocess.DEVNULL
                       )
        )

        self.process2 = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(self.frame_width, self.frame_height), framerate = self.framerate)
            .output(self.video_file2, vcodec='libx264', pix_fmt='yuv420p') # Using h264 codec for mp4
            .global_args('-nostats', '-loglevel', 'error')
            .overwrite_output()
            .run_async(pipe_stdin=True,
                       pipe_stdout=subprocess.DEVNULL,
                       pipe_stderr=subprocess.DEVNULL
                       )
        )

    def record_frame(self, frame, frame_no_gaze, gaze):
        self.process1.stdin.write(frame.tobytes())
        self.process2.stdin.write(frame_no_gaze.tobytes())

        if gaze is None:
            gaze = np.array([np.nan, np.nan], dtype=float)
        self.gazes.append(gaze)

    def end_recording(self):
        # Clean up and finish writing to video
        self.process1.stdin.close()
        self.process1.wait()
        self.process2.stdin.close()
        self.process2.wait()

        print(f'Recording Terminated.')
        print(f"Videos saved to {self.video_file1} and {self.video_file2}")

        # Finish writing gaze coordinate information
        np.save(self.gaze_file, np.array(self.gazes, dtype=object))
        print(f'Gaze coordinates saved to {self.gaze_file}')