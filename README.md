# Aria Control
This is a Python package that provides an easy-to-use interface for interacting with Project Aria smart glasses.

## Installation

### Install Environment
#### Option 1
You can use the existing environment.yml file to install the environment needed by
```
conda env create -f environment.yml
conda activate aria-control
```

#### Option 2
If you prefer to manually install packages needed, 
1. create an environment with python 3.10 by
    ```
    conda create -n your_env_name python=3.10
    conda activate your_env_name
    ```
2. install project aria SDK by
    ```
    python -m pip install --upgrade pip
    python -m pip install projectaria_client_sdk --no-cache-dir
    ```

#### Post-installation check
```
aria-doctor  # detect and resolve common issues connecting and streaming from the glasses
```

### Install This Package
1. clone this repo
2. navigate to the root folder of this repo (It should show `(env) username@YOURCOMP aria-control` in your terminal)
3. `pip install -e .`

## Run Streaming, Recording, or Gaze Inferring
### Connect your device via wifi
1. run `aria auth pair` - open the Aria App on the phone - click "Approve".
2. make sure that your computer and the glasses are on the **same network** (or accessible). Then run
    ```
    aria streaming start --interface wifi --device-ip YOUR_IP
    ```
    to start streaming.

### Run your script
Feel free to refer to the examples provided under `./examples`.

**Make sure you have a config file for your glasses under your working directory.** You can copy the template from `aria_glasses/default_config.yaml`. If you put it under the same directory of the file you'll run and name it with `config.yaml`, no need to pass in this argument.
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
