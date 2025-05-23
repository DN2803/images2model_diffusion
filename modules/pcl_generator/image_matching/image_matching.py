import os
import shutil
from pathlib import Path
from pprint import pprint

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    GeometricVerification,
    Quality,
    TileSelection,
    Timer,
    logger
)
from modules import extractors, matchers
from modules.extractors.extractor_base import extractor_loader
from modules.matchers.matcher_base import matcher_loader
from .pairs_generator import PairsGenerator
from utils.image import ImageList


def make_correspondence_matrix(matches: np.ndarray) -> np.ndarray:
    kpts_number = matches.shape[0]
    n_tie_points = np.arange(kpts_number).reshape((-1, 1))
    matrix = np.hstack((n_tie_points, matches.reshape((-1, 1))))
    correspondences = matrix[~np.any(matrix == -1, axis=1)]
    return correspondences


def get_pairs_from_file(pair_file: Path) -> list:
    pairs = []
    with open(pair_file, "r") as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            im1, im2 = line.strip().split(" ", 1)
            pairs.append((im1, im2))
    return pairs


class ImageMatching:
    """
    ImageMatching class for performing image matching and feature extraction.

    Methods:
        __init__(self, imgs_dir, output_dir, matching_strategy, local_features, matching_method, retrieval_option=None, pair_file=None, overlap=None, existing_colmap_model=None, custom_config={})
            Initializes the ImageMatching class.
        generate_pairs(self, **kwargs) -> Path:
            Generates pairs of images for matching.
        rotate_upright_images(self)
            Rotates upright images.
        extract_features(self) -> Path:
            Extracts features from the images.
        match_pairs(self, feature_path, try_full_image=False) -> Path:
            Matches pairs of images.
        rotate_back_features(self, feature_path)
            Rotates back the features.

    """

    default_conf_general = {
        "quality": Quality.HIGH,
        "tile_selection": TileSelection.NONE,
        "geom_verification": GeometricVerification.PYDEGENSAC,
        "output_dir": "output",
        "tile_size": [2048, 1365],
        "tile_overlap": 0,
        "force_cpu": False,
        "do_viz": False,
        "fast_viz": True,
        "hide_matching_track": True,
        "do_viz_tiles": False,
    }
    # pair_file=pair_file,
    # retrieval_option=retrieval_option,
    # overlap=overlap,
    # existing_colmap_model=existing_colmap_model,

    def __init__(
        # TODO: add default values for not necessary parameters
        self,
        imgs_dir: Path,
        output_dir: Path,
        matching_strategy: str,
        local_features: str,
        matching_method: str,
        retrieval_option: str = None,
        pair_file: Path = None,
        overlap: int = None,
        existing_colmap_model: Path = None,
        custom_config: dict = {},
    ):
        """
        Initializes the ImageMatching class.

        Parameters:
            imgs_dir (Path): Path to the directory containing the images.
            output_dir (Path): Path to the output directory for the results.
            matching_strategy (str): The strategy for generating pairs of images for matching.
            local_features (str): The method for extracting local features from the images.
            matching_method (str): The method for matching pairs of images.
            retrieval_option (str, optional): The retrieval option for generating pairs of images. Defaults to None.
            pair_file (Path, optional): Path to the file containing custom pairs of images. Required when 'retrieval_option' is set to 'custom_pairs'. Defaults to None.
            overlap (int, optional): The overlap between tiles. Required when 'retrieval_option' is set to 'sequential'. Defaults to None.
            existing_colmap_model (Path, optional): Path to the existing COLMAP model. Required when 'retrieval_option' is set to 'covisibility'. Defaults to None.
            custom_config (dict, optional): Custom configuration settings. Defaults to {}.

        Raises:
            ValueError: If the 'overlap' option is required but not provided when 'retrieval_option' is set to 'sequential'.
            ValueError: If the 'pair_file' option is required but not provided when 'retrieval_option' is set to 'custom_pairs'.
            ValueError: If the 'pair_file' does not exist when 'retrieval_option' is set to 'custom_pairs'.
            ValueError: If the 'existing_colmap_model' option is required but not provided when 'retrieval_option' is set to 'covisibility'.
            ValueError: If the 'existing_colmap_model' does not exist when 'retrieval_option' is set to 'covisibility'.
            ValueError: If the image folder is empty or contains only one image.

        Returns:
            None
        """
        self.image_dir = Path(imgs_dir)
        self.output_dir = Path(output_dir)
        self.matching_strategy = matching_strategy
        self.retrieval_option = retrieval_option
        self.local_features = local_features
        self.matching_method = matching_method
        self.pair_file = Path(pair_file) if pair_file else None
        self.overlap = overlap
        self.existing_colmap_model = existing_colmap_model

        if self.output_dir.exists():
            custom_config["general"]["output_dir"] = self.output_dir
        # Merge default and custom config
        self.custom_config = custom_config
        self.custom_config["general"] = {
            **self.default_conf_general,
            **custom_config["general"],
        }
        print("Custom config:", self.custom_config["general"])
        # Check that parameters are valid
        if retrieval_option == "sequential":
            if overlap is None:
                raise ValueError(
                    "'overlap' option is required when 'strategy' is set to sequential"
                )
        elif retrieval_option == "custom_pairs":
            if self.pair_file is None:
                raise ValueError(
                    "'pair_file' option is required when 'strategy' is set to custom_pairs"
                )
            else:
                if not self.pair_file.exists():
                    raise ValueError(f"File {self.pair_file} does not exist")
        elif retrieval_option == "covisibility":
            if self.existing_colmap_model is None:
                raise ValueError(
                    "'existing_colmap_model' option is required when 'strategy' is set to covisibility"
                )
            else:
                if not self.existing_colmap_model.exists():
                    raise ValueError(
                        f"File {self.existing_colmap_model} does not exist"
                    )

        # Initialize ImageList class
        self.image_list = ImageList(imgs_dir)
        images = self.image_list.img_names
        if len(images) == 0:
            raise ValueError(f"Image folder empty. Supported formats: {self.image_ext}")
        elif len(images) == 1:
            raise ValueError("Image folder must contain at least two images")

        # Initialize output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractor
        try:
            Extractor = extractor_loader(extractors, self.local_features)
        except AttributeError:
            raise ValueError(
                f"Invalid local feature extractor. {self.local_features} is not supported."
            )
        self._extractor = Extractor(self.custom_config)

        # Initialize matcher
        try:
            Matcher = matcher_loader(matchers, self.matching_method)
        except AttributeError:
            raise ValueError(
                f"Invalid matcher. {self.matching_method} is not supported."
            )
        if self.matching_method == "lightglue":
            self._matcher = Matcher(
                local_features=self.local_features, config=self.custom_config
            )
        else:
            self._matcher = Matcher(self.custom_config)
            # self._matcher = Matcher(
            #     self.local_features, config=self.custom_config
            # )

        # Print configuration
        logger.info("Running image matching with the following configuration:")
        logger.info(f"  Image folder: {self.image_dir}")
        logger.info(f"  Output folder: {self.output_dir}")
        logger.info(f"  Number of images: {len(self.image_list)}")
        logger.info(f"  Matching strategy: {self.matching_strategy}")
        logger.info(f"  Image quality: {self.custom_config['general']['quality'].name}")
        logger.info(
            f"  Tile selection: {self.custom_config['general']['tile_selection'].name}"
        )
        logger.info(f"  Feature extraction method: {self.local_features}")
        logger.info(f"  Matching method: {self.matching_method}")
        logger.info(
            f"  Geometric verification: {self.custom_config['general']['geom_verification'].name}"
        )
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")

    @property
    def img_names(self):
        return self.image_list.img_names

    def generate_pairs(self, **kwargs) -> Path:
        """
        Generates pairs of images for matching.

        Returns:
            Path: The path to the pair file containing the generated pairs of images.
        """
        if self.pair_file is not None and self.matching_strategy == "custom_pairs":
            if not self.pair_file.exists():
                raise FileExistsError(f"File {self.pair_file} does not exist")

            pairs = get_pairs_from_file(self.pair_file)
            self.pairs = [
                (self.image_dir / im1, self.image_dir / im2) for im1, im2 in pairs
            ]

        else:
            pairs_generator = PairsGenerator(
                self.image_list.img_paths,
                self.pair_file,
                self.matching_strategy,
                self.retrieval_option,
                self.overlap,
                self.image_dir,
                self.output_dir,
                self.existing_colmap_model,
                **kwargs,
            )
            self.pairs = pairs_generator.run()

        return self.pair_file

    

    def extract_features(self) -> Path:
        """
        Extracts features from the images using the specified local feature extraction method.

        Returns:
            Path: The path to the directory containing the extracted features.

        Raises:
            ValueError: If the local feature extraction method is invalid or not supported.

        """
        logger.info(f"Extracting features with {self.local_features}...")
        logger.info(f"{self.local_features} configuration: ")
        pprint(self.custom_config["extractor"])

        # Extract features
        for img in tqdm(self.image_list):
            feature_path = self._extractor.extract(img)

        torch.cuda.empty_cache()
        logger.info("Features extracted!")

        return feature_path

    def match_pairs(self, feature_path: Path, try_full_image: bool = False) -> Path:
        """
        Matches features using a specified matching method.

        Args:
            feature_path (Path): The path to the directory containing the extracted features.
            try_full_image (bool, optional): Whether to try matching the full image. Defaults to False.

        Returns:
            Path: The path to the directory containing the matches.

        Raises:
            ValueError: If the feature path does not exist.
        """
        timer = Timer(log_level="debug")

        logger.info(f"Matching features with {self.matching_method}...")
        logger.info(f"{self.matching_method} configuration: ")
        pprint(self.custom_config["matcher"])
        # Check that feature_path exists
        feature_path = Path(feature_path)
        if not feature_path.exists():
            raise ValueError(f"Feature path {feature_path} does not exist")

        # Define matches path
        matches_path = feature_path.parent / "matches.h5"

        # Match pairs
        logger.info("Matching features...")
        logger.info("")
        for i, pair in enumerate(tqdm(self.pairs)):
            name0 = pair[0].name if isinstance(pair[0], Path) else pair[0]
            name1 = pair[1].name if isinstance(pair[1], Path) else pair[1]
            im0 = self.image_dir / name0
            im1 = self.image_dir / name1

            logger.debug(f"Matching image pair: {name0} - {name1}")

            # Run matching
            self._matcher.match(
                feature_path=feature_path,
                matches_path=matches_path,
                img0=im0,
                img1=im1,
                try_full_image=try_full_image,
            )
            timer.update("Match pair")

            # NOTE: Geometric verif. has been moved to the end of the matching process

        # TODO: Clean up features with no matches

        torch.cuda.empty_cache()
        timer.print("matching")

        return matches_path

