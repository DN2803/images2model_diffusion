from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from utils.mvs.preprocess import *

class MVSDataset(Dataset):
    def __init__(self, datapath, nviews, ndepths=192, interval_scale=1.06):
        super(MVSDataset, self).__init__()
        self.datapath = datapath  # path to the scan folder
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        pair_file = os.path.join(self.datapath, "pair.txt")
        with open(pair_file) as f:
            num_viewpoint = int(f.readline())
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().strip())
                src_views = [int(x) for x in f.readline().strip().split()[1::2]]
                metas.append((ref_view, src_views))
        print("Loaded {} pairs from {}".format(len(metas), pair_file))
        return metas

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        img = np.array(img, dtype=np.float32) / 255.
        assert img.shape[:2] == (1200, 1600)
        img = img[:-16, :]
        return img

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        proj_matrices = []
        depth_values = None

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, "images", f"{vid:08d}.jpg")
            cam_filename = os.path.join(self.datapath, "cams", f"{vid:08d}_cam.txt")

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(cam_filename)

            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            imgs.append(img)
            proj_matrices.append(proj_mat)

            if i == 0:
                depth_values = np.arange(
                    depth_min,
                    depth_interval * (self.ndepths - 0.5) + depth_min,
                    depth_interval,
                    dtype=np.float32
                )

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])  # NCHW
        proj_matrices = np.stack(proj_matrices)

        return {
            "imgs": imgs,
            "proj_matrices": proj_matrices,
            "depth_values": depth_values,
            "filename": f"{ref_view:08d}"
        }
