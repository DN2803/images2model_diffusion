from abc import ABC, abstractmethod
import h5py
import numpy as np
from collections import defaultdict


class MatcherBase(ABC):
    def __init__(self):
        pass
    def loader(self, weight): 
        pass 
    @abstractmethod
    def matching(image_pair):
        pass
    @staticmethod
    def save_to_h5(matched_pairs, feature_path, match_path):
   
        """
        Lưu keypoints vào `feature_path.h5` và matches vào `match_path.h5`,
        đồng thời nhóm keypoints theo từng ảnh và cập nhật lại index trong matches.
        """

        # Tạo tập hợp keypoints cho từng ảnh
        image_keypoints = defaultdict(list)
        image_keypoints_indices = {}

        # Gộp keypoints từ tất cả các cặp ảnh
        for (image0, image1), match_data in matched_pairs.items():
            keypoints0 = match_data["keypoints0"]
            keypoints1 = match_data["keypoints1"]
            matches = match_data["matches"]

            # Lọc matches hợp lệ (loại bỏ match có -1)
            valid_indices = matches != -1
            valid_matches = np.column_stack((np.arange(len(matches))[valid_indices], matches[valid_indices]))

            # Thêm keypoints vào danh sách nếu chưa có
            for idx, kpt in enumerate(keypoints0):
                kpt_tuple = tuple(kpt.tolist())
                if kpt_tuple not in image_keypoints_indices:
                    image_keypoints_indices[(image0, kpt_tuple)] = len(image_keypoints[image0])
                    image_keypoints[image0].append(kpt)

            for idx, kpt in enumerate(keypoints1):
                kpt_tuple = tuple(kpt.tolist())
                if kpt_tuple not in image_keypoints_indices:
                    image_keypoints_indices[(image1, kpt_tuple)] = len(image_keypoints[image1])
                    image_keypoints[image1].append(kpt)

        # Cập nhật index của matches theo keypoints mới
        updated_matches = {}
        for (image0, image1), match_data in matched_pairs.items():
            keypoints0 = match_data["keypoints0"]
            keypoints1 = match_data["keypoints1"]
            matches = match_data["matches"]

            # Lọc matches hợp lệ
            valid_indices = matches != -1
            valid_matches = np.column_stack((np.arange(len(matches))[valid_indices], matches[valid_indices]))

            remapped_matches = np.array([
                [
                    image_keypoints_indices[(image0, tuple(keypoints0[old0]))],
                    image_keypoints_indices[(image1, tuple(keypoints1[old1]))]
                ]
                for old0, old1 in valid_matches
            ])

            updated_matches[(image0, image1)] = remapped_matches

        # Lưu keypoints vào feature.h5
        with h5py.File(feature_path, "w") as f_feat:
            for image, keypoints in image_keypoints.items():
                grp = f_feat.create_group(image)
                grp.create_dataset("keypoints", data=np.array(keypoints))

        # Lưu matches vào match.h5
        with h5py.File(match_path, "w") as f_match:
            for (image0, image1), matches in updated_matches.items():
                pair_name = f"{image0},{image1}"
                grp = f_match.create_group(pair_name)
                grp.create_dataset("matches", data=matches)

        print(f"✅ Keypoints đã gộp và lưu vào {feature_path}")
        print(f"✅ Matches đã ánh xạ lại index và lưu vào {match_path}")