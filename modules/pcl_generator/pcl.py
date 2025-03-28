import os 
import open3d as o3d

import numpy as np
import pycolmap 
import enlighten
from pathlib import Path
import yaml


from modules.matcher.diff_glue import DiffGlue

from modules.pcl_generator.image_matching.matcher import ImageMatcher
from utils.io.h5_to_db import export_to_colmap

import argparse



class PCL():
    def __init__(self, images_dir, output_dir):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.database_path = self.output_dir / "database.db"

        self.feature_path = self.output_dir / "feature.h5"
        self.match_path = self.output_dir / "match.h5"

        with open("config/camera_options.yaml", "r") as file:
            self.camera_options = yaml.safe_load(file)


    def generate(self):

        # Generate image mathching 
        images_matching = ImageMatcher()

        image_pairs = images_matching.generate_pairs(image_path=self.images_dir, method="sequential")
        matched_pairs = {}
        matcher = DiffGlue()
        opt = argparse.Namespace()
        opt.resize = [-1]
        opt.nms_radius = 3
        opt.keypoint_threshold =0.005
        opt.max_keypoints = 2048
        for image_pair in image_pairs: 
            # Feature Matching
            pred = matcher.matching(self.images_dir, image_pair, opt)
            matched_pairs[image_pair] = {
                "keypoints0": pred['keypoints0'][0].cpu().numpy(),
                "keypoints1": pred['keypoints1'][0].cpu().numpy(),
                "matches": pred['matches0'][0].cpu().numpy(),
                "confidence": pred['matching_scores0'][0].cpu().numpy()
            }
        matcher.save_to_h5(matched_pairs, self.feature_path, self.match_path)
        export_to_colmap(
            img_dir=self.images_dir,
            feature_path=self.feature_path,   # Đường dẫn đến file chứa keypoints
            match_path=self.match_path,       # Đường dẫn đến file chứa matches
            database_path=self.database_path, # Database COLMAP
            camera_options=self.camera_options
        )
        output = self.output_dir + '/mvs'

        num_images = pycolmap.Database(self.database_path).num_images
        with enlighten.Manager() as manager:
            with manager.counter(total=num_images, desc="Images registered:") as pbar:
                pbar.update(0, force=True)
                recs = pycolmap.incremental_mapping(
                    self.database_path,
                    self.images_dir,
                    output,
                    initial_image_pair_callback=lambda: pbar.update(2),
                    next_image_callback=lambda: pbar.update(1),
                )
        reconstruction = pycolmap.Reconstruction(self.output_dir)
        reconstruction.write_text(self.output_dir)  # text format
        print("Result is saving in {}".format(self.output_dir+"pcl.ply"))
        reconstruction.export_PLY("pcl.ply")  # PLY format

    