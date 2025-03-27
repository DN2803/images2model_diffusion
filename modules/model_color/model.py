from abc import ABC, abstractmethod

class MeshBase(ABC): 
    def __init__(self):
        pass
    @abstractmethod
    def reconstruct(self, pcl_path):
        pass 

