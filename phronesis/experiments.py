from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

def timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")

class Experiment:
    def __init__(self, defaults=None):
        #defaults = OmegaConf.create(defaults)
        self.config = OmegaConf.merge(OmegaConf.from_cli(), defaults)
        self.config.run = OmegaConf.create()
        self.config.run.run_id = timestamp()

        self.project_path = Path(self.config.project.path)
        self.run_path = self.project_path_ / self.config.run.run_id
        self.data_path = self.run_path / 'data'
        self.logs_path = self.run_path / 'logs'
        self.samples_path = self.run_path / 'samples'
        self.checkpoints_path = self.run_path / 'checkpoints'

        OmegaConf.save(self.config, self.run_path / 'config.yaml')

    def run(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()