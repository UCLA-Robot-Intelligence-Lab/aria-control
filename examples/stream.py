import os
import argparse
from aria_glasses import AriaGlasses
from aria_glasses.utils.general import quit_keypress
from datetime import datetime
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Specify the path to the configuration file",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        help="Specify the folder name to save the recordings",
    )

    return parser.parse_args()


if __name__ == "__main__":

    # set up config
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = current_dir + '/' + args.config_path
    
    # initialize Aria Glasses
    glasses = AriaGlasses(config_path)

    # set up the live streaming and recording
    glasses.start_streaming(live_view=True)

    while not quit_keypress():
        gaze = glasses.infer_gaze(mode='in_rgb')
        rgb_image = glasses.get_frame_image('rgb')

        if rgb_image is not None:

            glasses.view_frame(rgb_image, gaze, with_gaze=True, with_text=True)

    glasses.stop_streaming()
