import os
from types import SimpleNamespace

import yaml

class ConfigLoader():
    def __init__(self, config_path):
        default_config_path = 'default_config.yml'
        with open(default_config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        with open(config_path, 'r') as file:
            self.config = {**self.config, **yaml.safe_load(file)}
        
    def get_config(self):
        return SimpleNamespace(**self.config)