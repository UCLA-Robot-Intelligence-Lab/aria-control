import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    '''
    Manages configuration settings for the AriaGlasses package.
    
    This class handles loading, updating, and saving configuration settings
    from a YAML file. It supports nested configuration using dot notation.
    '''
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        '''
        Initialize the ConfigManager.
        
        Args:
            config_path (Optional[str]): Path to the config file. If None,
                                       searches for config files automatically.
        '''
        self.config_path = config_path or get_config_path()
        self.is_default = str(self.config_path).endswith("default_config.yaml")
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        '''
        Load configuration from YAML file.
        
        Returns:
            Dict[str, Any]: Loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid YAML
        '''
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        '''
        Get a configuration value using dot notation.
        
        Args:
            key (str): Configuration key in dot notation (e.g., "device.ip")
            default (Any, optional): Default value if key not found
            
        Returns:
            Any: Configuration value or default if not found
        '''
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
                
        return value
        
    def update(self, key: str, value: Any) -> None:
        '''
        Update a configuration value using dot notation.
        
        Args:
            key (str): Configuration key in dot notation
            value (Any): New value to set
            
        Example:
            >>> config.update("device.ip", "192.168.1.100")
        '''
        if self.is_default:
            raise RuntimeError("Cannot update default configuration. Create a custom config file instead.")
            
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def save(self) -> None:
        '''
        Save current configuration to file.
        
        Raises:
            IOError: If file cannot be written
            RuntimeError: If trying to save default configuration
        '''
        if self.is_default:
            raise RuntimeError("Cannot save default configuration. Create a custom config file instead.")
            
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def create_custom_config(self, save_path: Optional[str] = None) -> str:
        '''
        Create a custom config file based on the default configuration.
        
        Args:
            save_path (Optional[str]): Path where to save the custom config file.
                                     If None, saves to ~/.aria_glasses.yaml
            
        Returns:
            str: Path to the created config file
            
        Raises:
            FileExistsError: If file already exists at save_path
        '''
        if save_path is None:
            # Default to user's home directory
            save_path = os.path.expanduser("~/.aria_glasses.yaml")
        
        if os.path.exists(save_path):
            raise FileExistsError(f"File already exists at {save_path}")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        return save_path
    


def get_config_path() -> str:
    '''
    Get the path to the config file.
    
    The function checks for config files in the following order:
    1. Current directory (aria_glasses.yaml)
    2. Home directory (~/.aria_glasses.yaml)
    3. Default config from package
    
    Returns:
        str: Path to the config file
    '''
    # Check current directory
    current_dir_config = "aria_glasses.yaml"
    if os.path.exists(current_dir_config):
        return current_dir_config
        
    # Check home directory
    home_config = os.path.expanduser("~/.aria_glasses.yaml")
    if os.path.exists(home_config):
        return home_config
        
    # Return default config path (now in the root aria_glasses directory)
    return str(Path(__file__).parent.parent / "default_config.yaml")