import os
import sys
import cv2
import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
import pkg_resources
from datetime import datetime

# from .utils.config_manager import ConfigManager
from aria_glasses.utils.config_manager import ConfigManager
from aria_glasses.utils.general import *
from aria_glasses.utils.streaming import *
from aria_glasses.eyetracking.inference import infer
from aria_glasses.utils.recording import VideoRecorder, GazeRecorder

import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.calibration import device_calibration_from_json_string, distort_by_calibration, get_linear_camera_calibration
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.mps import EyeGaze, get_eyegaze_point_at_depth



class AriaGlasses:
    '''
    AriaGlasses class for interacting with Meta's Project Aria glasses.
    
    This class encapsulates functionality for connecting to Aria glasses,
    streaming data, recording sessions, and processing gaze data.
    '''
    def __init__(self, 
                 config_path: Optional[str] = pkg_resources.resource_filename('aria_glasses', 'default_config.yaml')
                 ) -> None:
        '''
        Initialize the streaming client with device IP, config path, and visualization settings.
        
        Args:
            config_path (Optional[str]): Path to configuration file. If None, will use default settings.
        '''
        self.config_manager = ConfigManager(config_path)
        
        self.device_ip = self.config_manager.get('device-ip', None)
        print(self.device_ip)
        if not self.device_ip:
            raise ValueError("Please specify your Aria device ip in the config file")

        if sys.platform.startswith("linux"):
            update_iptables()

        self.log_level = self.config_manager.get('log_level')
        self._setup_logging()

        self.rgb_cam_res = self.config_manager.get('rgb_cam_res', [1408, 1408])
        self.et_cam_res = self.config_manager.get('et_cam_res', [240, 320])

        self.connected = False
        self.stream_active = False
        self.record_active = False

        self.rgb_stream_label = self.config_manager.get('streaming.rgb_stream_label', 'camera-rgb')

        # self.gaze = np.array(([0, 0]), dtype=np.float32)
        self.gaze = None
        self._setup_gaze_inference()

    def _setup_logging(self) -> None:
        if self.log_level=='Info':
            aria.set_log_level(aria.Level.Info)
        elif self.log_level=='Debug':
            aria.set_log_level(aria.Level.Debug)
        elif self.log_level=='Trace':
            aria.set_log_level(aria.Level.Trace)

    def _setup_gaze_inference(self) -> None:
        model_weights = pkg_resources.resource_filename(
            'aria_glasses',
            'eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth'
        )
        model_config = pkg_resources.resource_filename(
            'aria_glasses',
            'eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml'
        )
        self.model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gaze_model = infer.EyeGazeInference(model_weights, model_config, self.model_device)

    def connect(self) -> bool:
        '''
        Connect to Aria glasses.
        '''
        try:
            self.device_client = aria.DeviceClient()
            client_config = aria.DeviceClientConfig()
            
            if self.device_ip:
                client_config.ip_v4_address = self.device_ip
                
            self.device_client.set_client_config(client_config)
            self.device = self.device_client.connect()
            
            self.connected = True
            print(f"Connected to Aria glasses at {self.device_ip}")

        except Exception as e:
            self.connected = False
            print(f"Aria connection failed: {e}")

    def start_streaming(self, live_view=True) -> bool:
        '''
        Start streaming data from Aria glasses. Subscribes to RGB and EyeTrack streams.
        '''
        if not self.connected:
            self.connect()
        
        if self.connected:
            try:
                streaming_manager = self.device.streaming_manager
                self.streaming_client = aria.StreamingClient()
                
                # device calibration
                sensors_calib_json = streaming_manager.sensors_calibration()
                self.device_calibration = device_calibration_from_json_string(sensors_calib_json)
                self.rgb_camera_calibration = self.device_calibration.get_camera_calib(self.rgb_stream_label)
                self.dist_calib = get_linear_camera_calibration(512, 512, 150, self.rgb_stream_label)

                # update subs_config
                self.data_types = self.config_manager.get('streaming.data_types', [])
                subs_config = self.streaming_client.subscription_config
                subs_config = update_subscription_config(subs_config, self.data_types)
                self.streaming_client.subscription_config = subs_config

                # create and attach observer
                self.observer = StreamingClientObserver()
                self.streaming_client.set_streaming_client_observer(self.observer)

                # start listening
                self.streaming_client.subscribe()
                self.stream_active = True
                print("Start listening to image data")

                if live_view:
                    self.stream_viewer = StreamViewer(self.config_manager)

            except Exception as e:
                self.stream_active = False
                print(f"Failed to start streaming: {e}")

    def stop_streaming(self) -> bool:
        '''
        Stop streaming data from Aria glasses.
        
        Returns:
            bool: True if streaming stopped successfully, False otherwise
        '''
        if not self.stream_active:
            print("Cannot stop streaming: Streaming is not active")
            return

        try:
            self.streaming_client.unsubscribe()
            self.stream_active = False

        except Exception as e:
            print(f"Failed to stop streaming: {e}")
    
    def start_recording(self, save_dir: str, natural=False) -> None:
        '''
        Start recording RGB frames and gaze coordinates data from Aria glasses.
        '''
        if not self.stream_active:
            print("Cannot start recording: Streaming is not active")
            return

        if save_dir is None:
            print(f"Please specify the saving folder.")
            return

        try:
            save_dir = str(save_dir)
            video_name = os.path.join(save_dir, 'rgb_cam')
            gaze_name = os.path.join(save_dir, 'raw_gaze')
            if natural:
                video_name = video_name + '_natural'
                gaze_name = gaze_name + '_natural'

            framerate = self.config_manager.get('streaming.framerate', 10)

            self.video_recorder = VideoRecorder(
                video_name=video_name,
                framerate=framerate
            )
            self.gaze_recorder = GazeRecorder(
                gaze_name=gaze_name
            )

            self.record_active = True
            print(f"Aria recording started============================================================")
        
        except Exception as e:
            self.record_active = False
            print(f"Failed to start recording: {e}")
            return
    
    def record_frame(self, image: np.ndarray, image_no_gaze: np.ndarray, gaze: np.ndarray) -> None:
        self.video_recorder.record_frame(image, image_no_gaze)
        self.gaze_recorder.record_frame(gaze)

    def view_frame(self, image: np.ndarray, gaze: np.ndarray, with_gaze=True, with_text=True) -> None:
        self.stream_viewer.update(image, gaze, with_gaze, with_text)

    def stop_recording(self) -> bool:
        '''
        Stop recording data from Aria glasses.
        '''
        if not self.record_active:
            print("Recording is not active.")
            return
        
        self.video_recorder.end_recording()
        self.gaze_recorder.end_recording()
        self.record_active = False

    def get_frame_image(self, camera_id: str = 'et') -> Optional[np.ndarray]:
        '''
        Get the latest frame from the specified camera.
        
        Args:
            camera_id (str): Camera identifier (e.g., 'et', 'rgb')
        '''
        if camera_id not in self.data_types:
            print(f"Camera ID {camera_id} not found in streaming frames. Please check if the camera is streaming.")
            return
        
        if camera_id == 'et':
            image = self.observer.images.get(aria.CameraId.EyeTrack)

        elif camera_id == 'rgb':
            image = self.observer.images.get(aria.CameraId.Rgb)
            # rotate by default
            if image is not None:
                image = np.rot90(image, -1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                del self.observer.images[aria.CameraId.Rgb]

        return image

    def infer_gaze(self, mode: str = '2d') -> np.ndarray:
        '''
        Process gaze data using the current frame.
        
        Args:
            mode (str): Gaze processing mode ('in_rgb' or 'raw')
                    'in_rgb' - 2D gaze vector reprojection with rotated angle aligning with rgb image
                    'raw' - 3D gaze point in CPF

        Returns:
            Array[np.ndarray]: Processed frame with gaze visualization

        Note:
            Details of function get_gaze_vector_reprojection: 
                https://github.com/facebookresearch/projectaria_tools/blob/7f6581c855c00be2c9f9f8c970e4dbe48aee85bd/projectaria_tools/core/mps/utils.py#L146
        '''
        if not self.stream_active:
            print("Cannot infer gaze: Streaming is not active")
            return

        et_image = self.get_frame_image('et')
        if et_image is None:
            # print("No eyetracking image available")
            # print(self.observer.images.get(aria.CameraId.EyeTrack))
            return
        
        et_image = torch.tensor(et_image)
        
        if np.median(et_image) < 10:
            # print("No gaze data available")
            return None

        with torch.no_grad():
            preds, lower, upper = self.gaze_model.predict(et_image)
            if self.model_device == 'cpu':
                preds = preds.detach().cpu().numpy() 
                lower = lower.detach().cpu().numpy()
                upper = upper.detach().cpu().numpy()

        eye_gaze = EyeGaze
        eye_gaze.yaw = preds[0][0]
        eye_gaze.pitch = preds[0][1]

        self.depth_m = self.config_manager.get('gaze.depth_m', 1.0)

        # print(f"{eye_gaze.yaw}, {eye_gaze.pitch, 3}, {self.depth_m}")
        # print(get_eyegaze_point_at_depth(eye_gaze.yaw, eye_gaze.pitch, self.depth_m))
        if mode == 'raw':
            self.gaze = get_eyegaze_point_at_depth(eye_gaze.yaw, eye_gaze.pitch, self.depth_m)      # left-to-right saccade, up-to-down saccade
            return self.gaze
        
        if mode == 'in_rgb':
            gaze_2d = get_gaze_vector_reprojection(
                eye_gaze,
                self.rgb_stream_label,
                self.device_calibration,
                self.rgb_camera_calibration,
                self.depth_m,
            )

            # adjust for 2d image alignment
            if gaze_2d.any() is None:
                # print("No gaze data available")
                return None
            else:
                x, y = gaze_2d
                rotated_x = self.rgb_cam_res[0] - y
                rotated_y = x
                self.gaze = np.array([rotated_x, rotated_y], dtype=np.float32)
                return self.gaze
    
    def get_undistorted_image(self, dist_image):
        undist_image = distort_by_calibration(dist_image, self.dist_calib, self.rgb_camera_calibration)
        return undist_image

    def get_rgb_camera_intrinsics_and_distortion(self) -> Tuple[np.ndarray, np.ndarray]:

        fx, fy = self.rgb_camera_calibration.get_focal_lengths()
        cx, cy = self.rgb_camera_calibration.get_principal_point()

        self.intrinsics = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
        self.distortion = self.rgb_camera_calibration.projection_params()[3:8]  # k1, k2, k3, p1, p2

        return self.intrinsics, self.distortion