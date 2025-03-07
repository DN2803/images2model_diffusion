from abc import ABC, abstractmethod

class Generator(ABC):  
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass