from transformers import pipeline
import torch

class DepthAnything(): 
    def __init__(self, model ="depth-anything/Depth-Anything-V2-Large-hf"):
        self.pipeline = pipeline(task="depth-estimation", model= model)

    def estimate_depth(self, image):
        depth = self.pipeline(image)["depth"]
        return depth
    
