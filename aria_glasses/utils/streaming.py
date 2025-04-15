import numpy as np
from projectaria_tools.core.sensor_data import ImageDataRecord
import aria.sdk as aria
import cv2
from aria_glasses.utils.general import display_text


class StreamingClientObserver:
    def __init__(self):
        self.images = {}

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image


def update_subscription_config(subs_config, data_types):
    # subscribe to specified streams
    type_mapping = {
                'rgb': aria.StreamingDataType.Rgb,
                'et': aria.StreamingDataType.EyeTrack,
                'slam': aria.StreamingDataType.Slam,
                'imu': aria.StreamingDataType.Imu,
                'audio': aria.StreamingDataType.Audio,
                'magneto': aria.StreamingDataType.Magneto,
                'baro': aria.StreamingDataType.Baro,
            }
    
    streaming_types = [type_mapping[type_name] for type_name in data_types if type_name in type_mapping]
    subs_config.subscriber_data_type = streaming_types[0]
    if len(streaming_types) > 1:
        for stype in streaming_types[1:]:
            subs_config.subscriber_data_type |= stype

    # set message queue size
    for type_name in data_types:
        if type_name in type_mapping:
            subs_config.message_queue_size[type_mapping[type_name]] = 1

    # set security options
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    subs_config.security_options = options

    return subs_config


def visualize_streaming(window, image, gaze_vis_params, gaze=np.array(([0, 0]), dtype=np.float32), with_gaze: bool = True, with_text: bool = True) -> None:
    '''
    Visualize the streaming data from specified cameras with optional gaze overlay and text annotations.
    
    Args:
        camera_ids (List[aria.CameraId], optional): List of camera IDs to visualize. 
            If None or empty, visualizes all available cameras. Defaults to None.
        with_gaze (bool, optional): Whether to overlay gaze information on the visualization. 
            Defaults to True.
        with_text (bool, optional): Whether to display text annotations on the visualization. 
            Defaults to True.
            
    Returns:
        None
    '''
    gp_color, gp_radius, gp_thickness = gaze_vis_params

    if with_gaze:
        cv2.circle(image, (int(gaze[0]), int(gaze[1])), gp_radius, gp_color, gp_thickness)

        if with_text:
            display_text(image, text=f'Gaze Coordinates: ({round(gaze[0], 4)}, {round(gaze[1], 4)})', position=(20, 90))
        cv2.imshow(window, image)

    else:
        cv2.imshow(window, image)