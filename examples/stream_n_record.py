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

    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = timestamp
    else:
        save_dir = args.save_dir

    data_pool = Path(__file__).resolve().parent.parent / "recordings"
    data_pool.mkdir(parents=True, exist_ok=True)
    trial_path = data_pool / save_dir
    trial_path.mkdir(parents=True, exist_ok=True)

    # initialize Aria Glasses
    glasses = AriaGlasses(config_path)

    # set up the live streaming and recording
    glasses.start_streaming(live_view=True)
    glasses.start_recording(trial_path)

    while not quit_keypress():
        gaze = glasses.infer_gaze(mode='in_rgb')
        rgb_image = glasses.get_frame_image('rgb')

        if rgb_image is not None:
            raw_rgb_image = rgb_image.copy()

            glasses.view_frame(rgb_image, gaze, with_gaze=True, with_text=True)
            glasses.record_frame(rgb_image, raw_rgb_image, gaze)

    glasses.stop_recording()
    glasses.stop_streaming()
