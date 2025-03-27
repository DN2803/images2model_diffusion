from abc import ABC, abstractmethod
import h5py


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
        Lưu keypoints vào `feature_path.h5` và matches vào `match_path.h5`
        để chuẩn bị import vào COLMAP.
        """
        # Lưu keypoints
        with h5py.File(feature_path, "w") as f_feat:
            for (image0, image1), match_data in matched_pairs.items():
                # Lưu keypoints cho từng ảnh
                if image0 not in f_feat:
                    f_feat.create_dataset(image0, data=match_data["keypoints0"])
                if image1 not in f_feat:
                    f_feat.create_dataset(image1, data=match_data["keypoints1"])

        # Lưu matches
        with h5py.File(match_path, "w") as f_match:
            for (image0, image1), match_data in matched_pairs.items():
                pair_name = f"{image0}-{image1}"
                grp = f_match.create_group(pair_name)
                grp.create_dataset("matches", data=match_data["matches"])
                grp.create_dataset("confidence", data=match_data["confidence"])

        print(f"Keypoints đã lưu vào {feature_path}")
        print(f"Matches đã lưu vào {match_path}")