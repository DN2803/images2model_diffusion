# buitl-in dependencies
import os

# 3rd-party dependencies
from PIL import Image

# project's dependencies
from models.generator import Generator
from models.depth_estimate.depth_anythingV2 import DepthAnything
from models.image_generate.zero123 import Zero123Plus

import numpy as np
from PIL import Image

class DepthImage(Generator):
    def __init__(self):
        self.model = DepthAnything()

    def generate(self, image, output_path):
        """
        Generate depth image from RGB image
        Args:
        - image: numpy array of RGB image
        - output_path: path to save depth image
        """
        
        depth_image = self.model.estimate_depth(image)
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
class DepthImages():
    def __init__(self, image_paths, depth_dir, color_dir):
        self.ori = image_paths
        self.target = [depth_dir, color_dir]
                
        os.makedirs(self.target[0], exist_ok=True)
        os.makedirs(self.target[1], exist_ok=True)
    def generator(self):
        color_paths=[]
        depth_paths=[]
        color_images=[]
        # normal_images=[]
        zero123 = Zero123Plus() 
        for i, image_path in enumerate(self.ori):
            image = Image.open(image_path)
            res_color_images, _ = zero123.generate(image)
            for img_tuple in res_color_images:
                if isinstance(img_tuple, tuple) and len(img_tuple) > 0:
                    color_images.append(img_tuple[0])  # Lấy ảnh từ tuple
                else:
                    color_images.append(img_tuple)  # Nếu không phải tuple, thêm trực tiếp 
            # normal_images.append(res_normal_images)

        depth = DepthImage()
        for i, color_img in enumerate(color_images):
            depth_path = f"{self.target[0]}/depth{i}.obj"
            color_path = f"{self.target[1]}/color{i}.png"
            color_img.save(color_path)
            depth.generate(color_img, depth_path)
            depth_paths.append(depth_path)
            color_paths.append(color_path)
        return (color_paths, depth_paths)     
        