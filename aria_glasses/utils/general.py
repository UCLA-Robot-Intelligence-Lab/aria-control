import signal
import subprocess
from contextlib import contextmanager
import cv2
from typing import Dict, Any, Tuple, List
import aria.sdk as aria



def read_gaze_vis_params(config_manager) -> Dict[str, Any]:
    '''
    Read gaze visualization parameters from configuration.
    '''
    color = tuple(config_manager.get('visualization.gaze_point_color', [0, 255, 0]))
    radius = config_manager.get('visualization.gaze_point_radius', 5)
    thickness = config_manager.get('visualization.gaze_point_thickness', 10)

    return [color, radius, thickness]

def read_vis_params(config_manager) -> Dict[str, Any]:
    '''
    Read visualization parameters from configuration.
    '''
    name = config_manager.get('visualization.window_name', 'Aria RGB')
    size = config_manager.get('visualization.window_size', [1024, 1024])
    position = config_manager.get('visualization.window_position', [50, 50])
    return name, size, position

def display_text(image, text: str, position, color=(0, 0, 255)):
    '''
    Display text on an image using OpenCV's putText function.
    
    Args:
        image: The image array to draw text on
        text (str): The text string to display
        position: Tuple of (x, y) coordinates where the text will be placed
        color (tuple): BGR color tuple for the text. Defaults to red (0, 0, 255)
        
    Example:
        display_text(frame, "Hello World", (20, 90))  # Red text
        display_text(frame, "Status", (20, 120), (0, 255, 0))  # Green text
    '''
    cv2.putText(
        img = image,
        text = text,
        org = position,
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        color = color,
        thickness = 3
    )


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def update_iptables() -> None:
    """
    Update firewall to permit incoming UDP connections for DDS
    """
    update_iptables_cmd = [
        "sudo",
        "iptables",
        "-A",
        "INPUT",
        "-p",
        "udp",
        "-m",
        "udp",
        "--dport",
        "7000:8000",
        "-j",
        "ACCEPT",
    ]
    print("Running the following command to update iptables:")
    print(update_iptables_cmd)
    subprocess.run(update_iptables_cmd)


@contextmanager
def ctrl_c_handler(signal_handler=None):
    class ctrl_c_state:
        def __init__(self):
            self._caught_ctrl_c = False

        def __bool__(self):
            return self._caught_ctrl_c

    state = ctrl_c_state()

    def _handler(sig, frame):
        state._caught_ctrl_c = True
        if signal_handler:
            signal_handler()

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handler)

    try:
        yield state
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)


def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC, 'q'
    return key == 27 or key == ord("q")

