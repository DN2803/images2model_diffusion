
def Generator(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generator = ABC()
    
    def generate(self, *args, **kwargs):
        return self._generator.generate(*args, **kwargs)