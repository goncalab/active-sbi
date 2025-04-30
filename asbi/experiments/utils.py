import sys
import yaml
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

