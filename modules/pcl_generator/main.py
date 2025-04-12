import os
import argparse
from dataclasses import dataclass, field, asdict
import yaml
import numpy as np
import open3d as o3d
import pycolmap
from tqdm import tqdm
from pathlib import Path

from modules.pcl_generator.image_matching.image_matching import ImageMatching
from utils.io.h5_to_db import export_to_colmap

import logging
import utils.timer as timer
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class GeneralConfig:
    matching_strategy: str = "bruteforce"
    pair_file: str = None
    retrieval: bool = False
    overlap: float = 0.5
    db_path: str = None
    upright: bool = False

@dataclass
class ExtractorConfig:
    name: str = "superpoint"
    max_keypoints: int = 10000

@dataclass
class MatcherConfig:
    name: str = "diffglue"

@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
default_config = Config()

class PCL:
    def __init__(self, images_dir, output_dir):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.pair_file_path = self.output_dir / "pairs.txt"
        self.database_path = self.output_dir / "database.db"
        self.feature_path = self.output_dir / "feature.h5"
        self.match_path = self.output_dir / "match.h5"

        with open("config/camera_options.yaml", "r") as file:
            self.camera_options = yaml.safe_load(file)

    def colmap_reconstruction(self, config=default_config):
        """Thực hiện quá trình tái tạo 3D bằng COLMAP."""
        logging.info("📸 Đang thực hiện COLMAP reconstruction...")
        print("Config:", config)

        if config.general["pair_file"] is not None:
            self.pair_file_path = self.images_dir / config.general["pair_file"]
            logging.info(f"📂 Đã tải file cặp ảnh từ: {self.pair_file_path}")
        img_matching = ImageMatching(
            imgs_dir=self.images_dir,
            output_dir=self.output_dir,
            matching_strategy=config.general.matching_strategy,
            local_features=config.extractor.name,
            matching_method=config.matcher.name,
            pair_file=self.pair_file_path,
            retrieval_option=config.general.retrieval,
            overlap=config.general.overlap,
            existing_colmap_model=config.general.db_path,
            custom_config=asdict(config),
        )

        pair_path = img_matching.generate_pairs()
        logging.info(f"📂 Đã tạo file cặp ảnh: {pair_path}")
        # Extract features
        feature_path = img_matching.extract_features()
        logging.info(f"📂 Đã tạo file đặc trưng: {feature_path}")
        # Matching
        match_path = img_matching.match_pairs(feature_path)
        logging.info(f"📂 Đã tạo file khớp: {match_path}")

        # If features have been extracted on "upright" images, this function bring features back to their original image orientation
        if config.general["upright"]:
            img_matching.rotate_back_features(feature_path)
            timer.update("rotate_back_features")
        logging.info("🗂️ Xuất dữ liệu từ h5 sang COLMAP database...")
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

        reconstruction = pycolmap.Reconstruction(self.output_dir / 'mvs')
        reconstruction.write_text(self.output_dir)  # Lưu dưới dạng text
        ply_path = self.output_dir / "pcl.ply"
        reconstruction.export_PLY(str(ply_path))  # Xuất PLY

        logging.info(f"✅ Kết quả đã lưu tại: {ply_path}")

    def generate(self):
        """Chạy toàn bộ pipeline."""
        self.colmap_reconstruction()
        self.save_ply()

    