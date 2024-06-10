import os
import json

# Load configuration constants from configuration file
script_dir = os.path.dirname(os.path.realpath(__file__))
relative_config_path = os.path.join(script_dir, '..', 'conf', 'config.json')
config_path = os.path.abspath(relative_config_path)
with open(config_path, 'r') as file:
    config = json.load(file)

if os.name == 'nt': # 'nt' stands for Windows
    DATA_FOLDER = config['win_data_folder']
    LOG_FILE = config['win_log_file']
elif os.name == 'posix': # 'posix' stands for Linux/Unix
    DATA_FOLDER = config['lin_data_folder']
    LOG_FILE = config['lin_log_file']
else:
    raise OSError("Unsupported operating system")