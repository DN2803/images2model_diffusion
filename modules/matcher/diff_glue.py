import numpy as np
import cv2
import torch


from models.matchers.models.matching import Matching

from modules.matcher.matcher_base import MatcherBase
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

class DiffGlue(MatcherBase): 
    def __init__(self):
        super().__init__()
    
    def matching (self, image_pair, opt):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints
            },
        }
        matching = Matching(config).eval().to(device)

        # Load the image pair.
        image0, inp0, scales0 = read_image(image_pair[0], device, opt.resize)
        image1, inp1, scales1 = read_image(image_pair[1], device, opt.resize)

        pred = matching({'image0': inp0, 'image1': inp1})
        
        # kpts0 = pred['keypoints0'][0].cpu().numpy()
        # kpts1 = pred['keypoints1'][0].cpu().numpy()
        # matches = pred['matches0'][0].cpu().numpy()
        # confidence = pred['matching_scores0'][0].cpu().numpy()

        return pred 

    