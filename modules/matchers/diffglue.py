import numpy as np
import cv2
import torch
from typing import Dict

from omegaconf import OmegaConf
from pathlib import Path

from models.matchers.models.diffglue_pipeline import DiffGluePipeline

from modules.matchers.matcher_base import FeaturesDict, MatcherBase

torch.set_grad_enabled(False)

import random

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    image = cv2.resize(image.astype('float32'), (w_new, h_new))

    inp = frame2tensor(image, device)
    return image, inp, scales

def features_2_sg(
    feats0: FeaturesDict, feats1: FeaturesDict, device: torch.device
) -> dict:
    # Merge feats0 and feats1 in a single dict
    data = {}
    data = {**data, **{k + "0": v for k, v in feats0.items()}}
    data = {**data, **{k + "1": v for k, v in feats1.items()}}
    if "feature_path0" in data.keys():
        del data["feature_path0"]
    if "feature_path1" in data.keys():
        del data["feature_path1"]
    if "im_path0" in data.keys():
        del data["im_path0"]
    if "im_path1" in data.keys():
        del data["im_path1"]
    data["image0"] = np.empty(data["image_size0"])
    data["image1"] = np.empty(data["image_size1"])
    data["image0"] = data["image0"][None]
    data["image1"] = data["image1"][None]

    # Add batch dimension
    data = {k: v[None] for k, v in data.items()}

    # Convert to tensor
    data = {
        k: torch.tensor(v, dtype=torch.float, device=device) for k, v in data.items()
    }

    # Add channel dimension if missing
    for i in range(2):
        s = data[f"image_size{i}"].cpu().numpy().astype(int).squeeze()
        data[f"image_size{i}"] = torch.Size((1, 1, s[0], s[1]))

    return data

def correspondence_matrix_from_matches0(
    kpts_number: int, matches0: np.ndarray
) -> np.ndarray:
    n_tie_points = np.arange(kpts_number).reshape((-1, 1))
    matrix = np.hstack((n_tie_points, matches0.reshape((-1, 1))))
    correspondences = matrix[~np.any(matrix == -1, axis=1)]

    return correspondences

class DiffGlueMatcher(MatcherBase): 
    default_config = {
        "name": "diffglue",
        "nms_radius": 3,
        "keypoint_threshold": 0.005,
        "max_keypoints": 2048,
        "local_features": "superpoint",
    }
    def __init__(self, config) -> None:
        """Initializes a DiffGlueMatcher object with the given options dictionary."""
        super().__init__(config)
        cfg = {**self.default_config, **self._config.get("matcher", {})}
        local_feat_name = cfg.get("local_features", "superpoint")

        if local_feat_name not in ["superpoint", "aliked"]:
            raise ValueError(f"Local feature '{local_feat_name}' is not supported. Please choose either 'superpoint' or 'aliked'.")

        default_conf = OmegaConf.create(DiffGluePipeline.default_conf)
        exper = Path("./models/matchers/weights/SP_DiffGlue.tar")

        if local_feat_name == "aliked":
            default_conf["local_features"] = "aliked"
            default_conf["input_dim"] = 128
            print(default_conf)
            exper = Path("./models/matchers/weights/ALIKED_DiffGlue.tar")

        self._matcher = DiffGluePipeline(default_conf).eval().cuda()  # load the matcher

        ckpt = exper
        ckpt = torch.load(str(ckpt), map_location="cpu")

        state_dict = ckpt["model"]
        dict_params = set(state_dict.keys())
        model_params = set(map(lambda n: n[0], self._matcher.named_parameters()))
        diff = model_params - dict_params
        if len(diff) > 0:
            state_dict = {k.replace('matcher.', 'matcher.net.'): v for k, v in state_dict.items()}
        self._matcher.load_state_dict(state_dict, strict=False)

       
    def _match_pairs(
        self,
        feats0: FeaturesDict,
        feats1: FeaturesDict,
        ) -> np.ndarray:
        set_seed(0)
        pred = {}

        data = features_2_sg(feats0, feats1, self._device)

        pred = self._matcher(data)
        pred = {
            k: v.cpu().numpy()
            for k, v in pred.items()
            if isinstance(v, torch.Tensor)
        }
        # kpts0 = pred['keypoints0'][0].cpu().numpy()
        # kpts1 = pred['keypoints1'][0].cpu().numpy()
        # matches = pred['matches0'][0].cpu().numpy()
        # confidence = pred['matching_scores0'][0].cpu().numpy()
        
        # Make correspondence matrix from matches0
        matches0 = pred["matches0"]
        kpts_number = feats0["keypoints"].shape[0]
        correspondences = correspondence_matrix_from_matches0(kpts_number, matches0)

        return correspondences