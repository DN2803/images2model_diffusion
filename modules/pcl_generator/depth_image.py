# buitl-in dependencies

# 3rd-party dependencies

import open3d as o3d
import numpy
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage import binary_erosion

from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from typing import Tuple

import torch

# project's dependencies

class DepthImage:
    def __init__(self, image = None):
        self.image = image
        self.height, self.width = self.image.shape[:2]
        self.processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
        self.model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    def create_depth_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        depth_image = np.where(self.depth == 0, np.nan, self.depth)
        depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
        image_o3d = o3d.geometry.Image(self.image)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            image_o3d, depth_o3d, 
            depth_scale=1000.0, 
            depth_trunc=1000.0, 
            convert_rgb_to_intensity=False)
        
        return rgbd_image
    def postprocess(rgb_img: Image.Image, normal_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        normal_vecs_pred = numpy.array(normal_img, dtype=numpy.float64) / 255.0 * 2 - 1
        alpha_pred = numpy.linalg.norm(normal_vecs_pred, axis=-1)

        is_foreground = alpha_pred > 0.6
        is_background = alpha_pred < 0.2
        structure = numpy.ones(
            (4, 4), dtype=numpy.uint8
        )

        is_foreground = binary_erosion(is_foreground, structure=structure)
        is_background = binary_erosion(is_background, structure=structure, border_value=1)

        trimap = numpy.full(alpha_pred.shape, dtype=numpy.uint8, fill_value=128)
        trimap[is_foreground] = 255
        trimap[is_background] = 0

        img_normalized = numpy.array(rgb_img, dtype=numpy.float64) / 255.0
        trimap_normalized = trimap.astype(numpy.float64) / 255.0

        alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
        foreground = estimate_foreground_ml(img_normalized, alpha)
        cutout = stack_images(foreground, alpha)

        cutout = numpy.clip(cutout * 255, 0, 255).astype(numpy.uint8)
        cutout = Image.fromarray(cutout)

        normal_vecs_pred = normal_vecs_pred / (numpy.linalg.norm(normal_vecs_pred, axis=-1, keepdims=True) + 1e-8)
        normal_vecs_pred = normal_vecs_pred * 0.5 + 0.5
        normal_vecs_pred = normal_vecs_pred * alpha[..., None] + 0.5 * (1 - alpha[..., None])
        normal_image_normalized = numpy.clip(normal_vecs_pred * 255, 0, 255).astype(numpy.uint8)

        return cutout, Image.fromarray(normal_image_normalized)
