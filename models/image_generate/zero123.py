# Description: Generate multiview images from a single image using zero123plus model

# 3rd-party dependencies
import cv2
import copy
import numpy
import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, ControlNetModel
from models.image_generate.matting_postprocess import postprocess


# project's dependencies
from utils.image import ImageUtils
from models.generator import Generator

class Zero123Plus(Generator):
    def __init__(self):

        ## load pipeline
        # Load the pipeline
        self.pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16
        )
        self.normal_pipeline = copy.copy(self.pipeline)
        self.normal_pipeline.add_controlnet(ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp12-normal-gen-v1", torch_dtype=torch.float16
        ), conditioning_scale=1.0)
        self.pipeline.to("cuda:0", torch.float16)
        self.normal_pipeline.to("cuda:0", torch.float16)


    def __rescale(self, single_res, input_image, ratio=0.95):
        # Rescale and recenter
        image_arr = numpy.array(input_image)
        ret, mask = cv2.threshold(numpy.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        side_len = int(max_size / ratio)
        padded_image = numpy.zeros((side_len, side_len, 4), dtype=numpy.uint8)
        center = side_len//2
        padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
        rgba = Image.fromarray(padded_image).resize((single_res, single_res), Image.LANCZOS)
        return rgba

    
    def __generate_multiview_images(self, image, rows=3, cols=2):
        """
        Generate 6 multiview images from a single image using zero123plus model.
        Args:
            image (Image): Input image.
            rows (int): num of image in rows.
            cols (int): num of image in cols.
        Returns:
            tuple: Tuple many of objecjt in type Image.Image.
        """
        
        # Run the pipeline
        cond = image
        # Optional: rescale input image if it occupies only a small region in input
        cond = self.__rescale(512, cond)
        # Generate 6 images
        genimg = self.pipeline(
            cond,
            prompt='', guidance_scale=4, num_inference_steps=75, width=640, height=960
        ).images[0]
        # Generate normal image
        # We observe that a higher CFG scale (4) is more robust
        # but with CFG = 1 it is faster and is usually good enough for normal image
        # You can adjust to your needs
        normalimg = self.normal_pipeline(
            cond, depth_image=genimg,
            prompt='', guidance_scale=4, num_inference_steps=75, width=640, height=960
        ).images[0]

        genimg, normalimg = postprocess(genimg, normalimg)


        genimgs = ImageUtils.split_image(genimg, rows, cols)
        normalimgs = ImageUtils.split_image(normalimg, rows, cols)
        return tuple([genimgs, normalimgs])
    
    def generate(self, image):
        return self.__generate_multiview_images(image)


