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
### Connect your device
If you connect glasses with computer via USB, there is one more step: run `aria auth pair` - open the Aria App on the phone - click "Approve".
If you connect to the glasses with WiFi, make sure that your computer and the glasses are on the **same network** (or accessible).

### Run your script
Feel free to refer to the examples provided under `./examples`.

## Frequent Q&A
1. Q: I cannot start streaming.
   A: Try to run `aria streaming start --interface wifi --device-ip YOUR_IP` and then run your file.
