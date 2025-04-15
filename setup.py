from setuptools import setup, find_packages

setup(
    name="aria_glasses",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        'aria_glasses': [
            'default_config.yaml',
            'eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml',
            'eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth',
        ],
    },
    install_requires=[
        'numpy',
        'opencv-python',
        'torch',
        'projectaria_client_sdk',
        'pyyaml',
    ],
)