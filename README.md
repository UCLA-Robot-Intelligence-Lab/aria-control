# Aria Control
This is a Python package that provides an easy-to-use interface for interacting with Project Aria smart glasses.

## Installation

### Clone the repo
```
# With HTTP
git clone https://github.com/xukristenyan/aria-control.git

# With SSH
git clone git@github.com:xukristenyan/aria-control.git
```

### Install
#### using uv
```
cd aria-control
uv sync
```

#### using conda
```
conda create -n gamma python=3.10
conda activate gamma
cd aria-control
pip install -e .
```

## Usage

### Connect your device via wifi
#### using uv
Make sure that your computer and the glasses are on the **same network** (or accessible).
```
# pair the glasses
uv run aria auth pair

# open the Aria app on the phone, click "Approve"

# start streaming
uv run aria streaming start --interface wifi --device-ip YOUR_IP
```

#### using conda
Make sure that your computer and the glasses are on the **same network** (or accessible).
```
# pair the glasses
aria auth pair

# open the Aria app on the phone, click "Approve"

# start streaming
aria streaming start --interface wifi --device-ip YOUR_IP
```

### Run your script
Feel free to refer to the examples provided under `./examples`.

**Make sure you have a config file for your glasses under your working directory.** You can copy the template from `aria_glasses/default_config.yaml`. If you put it under the same directory of the file you'll run and name it with `config.yaml`, no need to pass in this argument.

#### using uv
- visualize live streaming
    ```
    uv run examples/stream.py --config_path YOUR_CONFIG_PATH
    ```
- visualize live streaming & record
    ```
    uv run examples/stream_n_record.py --config_path YOUR_CONFIG_PATH
    ```
- check gaze data
    ```
    uv run examples/check_recorded_gaze.py
    ```

#### using conda
- visualize live streaming
    ```
    python examples/stream.py --config_path YOUR_CONFIG_PATH
    ```
- visualize live streaming & record
    ```
    python examples/stream_n_record.py --config_path YOUR_CONFIG_PATH
    ```
- check gaze data
    ```
    python examples/check_recorded_gaze.py
    ```