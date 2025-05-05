import sys
import yaml
import torch as th
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
     Load and parse the YAML configuration file
    """
    try:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error reading config file: {e}")
        sys.exit(1)

def get_device():
    """Get the device to use for training."""
    if th.cuda.is_available():
        device = th.device("cuda")
        print(f"Using GPU: {th.cuda.get_device_name(0)}")
    else:
        device = th.device("cpu")
        print("No GPU available, using CPU")
    return device
