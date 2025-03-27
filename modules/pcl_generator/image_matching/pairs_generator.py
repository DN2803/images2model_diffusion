import os
import itertools
import cv2
import numpy as np
from typing import List, Tuple

class PairsGenerator:
    def __init__(self, images_path: str, method: str = "all", max_pairs: int = None, **kwargs):
        """
        Khởi tạo bộ sinh cặp ảnh.
        
        Args:
            folder_path (str): Đường dẫn đến thư mục chứa ảnh.
            method (str): Cách chọn cặp ảnh ('all', 'sequential', 'random').
            max_pairs (int): Giới hạn số lượng cặp sinh ra.
        """
        self.images_path = images_path
        self.method = method
        self.max_pairs = max_pairs
        self.images = self._load_images()

    def _load_images(self) -> List[str]:
        """Load danh sách các ảnh trong folder."""
        images = [f for f in os.listdir(self.folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        images.sort()  # Để đảm bảo thứ tự cố định
        return images

    def generate_pairs(self) -> List[Tuple[str, str]]:
        """Sinh các cặp ảnh theo phương pháp chỉ định."""
        if self.method == "all":
            pairs = list(itertools.combinations(self.images, 2))  # Tạo tất cả cặp ảnh
        elif self.method == "sequential":
            pairs = [(self.images[i], self.images[i+1]) for i in range(len(self.images)-1)]
        elif self.method == "random":
            np.random.shuffle(self.images)
            pairs = list(itertools.combinations(self.images, 2))
        else:
            raise ValueError(f"Method '{self.method}' không hợp lệ. Chọn: 'all', 'sequential' hoặc 'random'.")

        if self.max_pairs:
            pairs = pairs[:self.max_pairs]  # Giới hạn số lượng cặp

        return pairs

    def filter_by_similarity(self, threshold: float = 0.7) -> List[Tuple[str, str]]:
        """Lọc các cặp ảnh dựa trên độ tương đồng (ORB + BFMatcher)."""
        pairs = self.generate_pairs()
        filtered_pairs = []
        
        for img1_name, img2_name in pairs:
            img1_path = os.path.join(self.images_path, img1_name)
            img2_path = os.path.join(self.images_path, img2_name)
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                continue

            # ORB feature matching
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            if des1 is None or des2 is None:
                continue

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            similarity = len(matches) / max(len(kp1), len(kp2))

            if similarity > threshold:
                filtered_pairs.append((img1_name, img2_name))

        return filtered_pairs
