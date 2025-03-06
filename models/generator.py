from abc import ABC, abstractmethod

def Generator (ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generator = ABC()
    @abstractmethod
    def generate(self, *args, **kwargs):
        return self._generator.generate(*args, **kwargs)
from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info