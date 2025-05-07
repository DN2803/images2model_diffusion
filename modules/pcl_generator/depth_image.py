# buitl-in dependencies
import logging
import os

# 3rd-party dependencies
from PIL import Image

# project's dependencies
from models.generator import Generator
from models.depth_estimate.depth_anythingV2 import DepthAnything
from models.image_generate.zero123 import Zero123Plus

import numpy as np
from PIL import Image
from pathlib import Path
import imageio.v3 as iio
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
        if isinstance(depth_image, Image.Image):
            depth_image = np.array(depth_image)

        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)
        self.save_pfm(depth_image, output_path)
        self.create_prob_pfm_from_depth(output_path)


    def save_pfm(self, image: np.ndarray, filename: Path):
        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')
        image = np.flipud(image)
        with open(filename, 'wb') as f:
            f.write(b'Pf\n')
            f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())
            f.write(b"-1.0\n")  # Little endian
            image.tofile(f)
    def create_prob_pfm_from_depth(depth_path):
        depth = iio.imread(depth_path, format='PFM')  # Đọc file .pfm
        prob = np.ones_like(depth, dtype=np.float32)  # Xác suất toàn bộ = 1.0

        prob_path = depth_path.replace("_init.pfm", "_prob.pfm")
        iio.imwrite(prob_path, prob, format='PFM')
        print(f"Saved {prob_path}")
class DepthImages():
    def __init__(self, images_dir: Path, depth_dir: Path):
        self.ori = images_dir
        self.target = depth_dir
                
        os.makedirs(self.target, exist_ok=True)

    def generator(self):
        depth = DepthImage()
        image_files = sorted(list(self.ori.glob("*")))
        for i, image_path in enumerate(image_files):
            img = Image.open(image_path).convert("RGB")
            basename = image_path.stem
            depth_path = self.target / f"{basename}_init.pfm"
            depth.generate(img, depth_path)

        logging.info(f"✅ Đã tạo ảnh độ sâu cho {len(image_files)} ảnh.")    
        