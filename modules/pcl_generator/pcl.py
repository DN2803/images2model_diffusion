import os
import argparse
import yaml
import numpy as np
import open3d as o3d
import pycolmap
from tqdm import tqdm
from pathlib import Path

from modules.matcher.diff_glue import DiffGlue
from modules.pcl_generator.image_matching.matcher import ImageMatcher
from utils.io.h5_to_db import export_to_colmap

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PCL:
    def __init__(self, images_dir, output_dir):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.database_path = self.output_dir / "database.db"
        self.feature_path = self.output_dir / "feature.h5"
        self.match_path = self.output_dir / "match.h5"

        with open("config/camera_options.yaml", "r") as file:
            self.camera_options = yaml.safe_load(file)

    def match_features(self):
        """Thực hiện tìm kiếm và matching keypoint giữa các ảnh."""
        logging.info("🔍 Đang thực hiện matching keypoints...")

        images_matching = ImageMatcher()
        image_pairs = images_matching.generate_pairs(images_path=self.images_dir, method="sequential")

        matched_pairs = {}
        matcher = DiffGlue()
        opt = argparse.Namespace()
        opt.resize = [-1]
        opt.nms_radius = 3
        opt.keypoint_threshold = 0.005
        opt.max_keypoints = 2048

        for image_pair in tqdm(image_pairs, desc="Matching Image Pairs"):
            pred = matcher.matching(self.images_dir, image_pair, opt)
            matched_pairs[image_pair] = {
                "keypoints0": pred['keypoints0'][0].cpu().numpy(),
                "keypoints1": pred['keypoints1'][0].cpu().numpy(),
                "matches": pred['matches0'][0].cpu().numpy(),
                "confidence": pred['matching_scores0'][0].cpu().numpy()
            }

        matcher.save_to_h5(matched_pairs, self.feature_path, self.match_path)

    def colmap_reconstruction(self):
        """Thực hiện quá trình tái tạo 3D bằng COLMAP."""
        logging.info("📸 Đang thực hiện COLMAP reconstruction...")

        export_to_colmap(
            img_dir=self.images_dir,
            feature_path=self.feature_path,
            match_path=self.match_path,
            database_path=self.database_path,
            camera_options=self.camera_options
        )

        output = self.output_dir / "mvs"
        num_images = pycolmap.Database(self.database_path).num_images

        logging.info(f"🖼️ Tổng số ảnh: {num_images}")

        pbar = tqdm(total=num_images, desc="Images Registered")
        recs = pycolmap.incremental_mapping(
            self.database_path,
            self.images_dir,
            output,
            initial_image_pair_callback=lambda: pbar.update(2),
            next_image_callback=lambda: pbar.update(1),
        )
        pbar.close()
        return recs

    def save_ply(self):
        """Xuất kết quả ra file PLY."""
        logging.info("📂 Đang lưu kết quả dưới dạng PLY...")

        reconstruction = pycolmap.Reconstruction(self.output_dir)
        reconstruction.write_text(self.output_dir)  # Lưu dưới dạng text
        ply_path = self.output_dir / "pcl.ply"
        reconstruction.export_PLY(str(ply_path))  # Xuất PLY

        logging.info(f"✅ Kết quả đã lưu tại: {ply_path}")

    def generate(self):
        """Chạy toàn bộ pipeline."""
        self.match_features()
        self.colmap_reconstruction()
        self.save_ply()

    