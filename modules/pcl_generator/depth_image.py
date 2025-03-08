# buitl-in dependencies

# 3rd-party dependencies
import PIL as Image


# project's dependencies
from models.generator import Generator
from models.depth_estimate.depth_anythingV2 import DepthAnything

import numpy as np
from PIL import Image

class DepthImage:
    def __init__(self, image=None):
        self.image = image
        if self.image is not None:
            self.height, self.width = self.image.shape[:2]  # Kiểm tra image trước khi truy cập shape

    @staticmethod
    def generate(image, output_path):
        """
        Generate depth image from RGB image
        Args:
        - image: numpy array of RGB image
        - output_path: path to save depth image
        """
        image = Image.fromarray(image)
        depth_image = DepthAnything.estimate_depth(image)
        H, W = depth_image.shape  # Kích thước ảnh

        with open(output_path, "w") as obj_file:
            for v in range(H): 
                for u in range(W):
                    z = depth_image[v, u]
                    if z == 0:
                        continue
                    x = u  # Sửa lỗi tuple
                    y = v  # Sửa lỗi tuple
                    obj_file.write(f"v {x} {y} {z}\n")

        return depth_image
