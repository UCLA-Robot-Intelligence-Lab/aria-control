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


# class VideoRecorder():
#     def __init__(self, video_name="rgb_cam", framerate=10, frame_height=1408, frame_width=1408):
        
#         self.video_file1 = video_name + ".mp4"
#         self.video_file2 = video_name + "_no_gaze.mp4"
#         self.framerate = int(framerate)
#         self.frame_height, self.frame_width, _ = frame_height, frame_width, 3

#         # Setup the ffmpeg pipe for video writing
#         self.process1 = (
#             ffmpeg
#             .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(self.frame_width, self.frame_height), framerate = self.framerate)
#             .output(self.video_file1, vcodec='libx264', pix_fmt='yuv420p') # Using h264 codec for mp4
#             .global_args('-nostats', '-loglevel', 'error')
#             .overwrite_output()
#             .run_async(pipe_stdin=True,
#                        pipe_stdout=subprocess.DEVNULL,
#                        pipe_stderr=subprocess.DEVNULL
#                        )
#         )

#         self.process2 = (
#             ffmpeg
#             .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(self.frame_width, self.frame_height), framerate = self.framerate)
#             .output(self.video_file2, vcodec='libx264', pix_fmt='yuv420p') # Using h264 codec for mp4
#             .global_args('-nostats', '-loglevel', 'error')
#             .overwrite_output()
#             .run_async(pipe_stdin=True,
#                        pipe_stdout=subprocess.DEVNULL,
#                        pipe_stderr=subprocess.DEVNULL
#                        )
#         )

#     def record_frame(self, frame, frame_no_gaze):
#         self.process1.stdin.write(frame.tobytes())
#         self.process2.stdin.write(frame_no_gaze.tobytes())

#     def end_recording(self):
#         # Clean up and finish writing to video
#         self.process1.stdin.close()
#         self.process1.wait()
#         self.process2.stdin.close()
#         self.process2.wait()

#         print(f'Recording Terminated.')
#         print(f"Videos saved to {self.video_file1} and {self.video_file2}")


class GazeRecorder():
    def __init__(self, gaze_name="raw_gaze"):
# TODO: add open file
        self.gaze_file = gaze_name + ".npy"
        print(f"gaze file is {self.gaze_file}")

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