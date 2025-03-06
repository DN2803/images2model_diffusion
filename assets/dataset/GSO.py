import os
import zipfile
import torch
import trimesh
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET


class GSO_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset cho Google Scanned Objects (GSO).
        :param root_dir: Thư mục chứa các thư mục đối tượng đã giải nén.
        :param transform: Transform áp dụng cho ảnh texture.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.object_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.object_dirs)

    def _read_sdf(self, sdf_path):
        """ Đọc file model.sdf để lấy đường dẫn model và texture. """
        try:
            tree = ET.parse(sdf_path)
            root = tree.getroot()
            mesh_uri = root.find(".//mesh/uri").text if root.find(".//mesh/uri") is not None else None
            texture_uri = root.find(".//material/script/uri").text if root.find(".//material/script/uri") is not None else None
            return mesh_uri, texture_uri
        except:
            return None, None

    def __getitem__(self, idx):
        obj_dir = self.object_dirs[idx]
        
        # Đọc file model.sdf
        sdf_path = os.path.join(obj_dir, "model.sdf")
        mesh_uri, texture_uri = self._read_sdf(sdf_path)

        # Đường dẫn đầy đủ
        mesh_path = os.path.join(obj_dir, mesh_uri) if mesh_uri else None
        texture_path = os.path.join(obj_dir, texture_uri) if texture_uri else None

        # Load model 3D
        mesh = trimesh.load_mesh(mesh_path) if mesh_path and os.path.exists(mesh_path) else None

        # Load texture nếu có
        texture = None
        if texture_path and os.path.exists(texture_path):
            texture = Image.open(texture_path).convert("RGB")
            if self.transform:
                texture = self.transform(texture)

        return {"mesh": mesh, "texture": texture, "mesh_path": mesh_path, "texture_path": texture_path}


# Sử dụng DataLoader
from torch.utils.data import DataLoader
from torchvision import transforms

# Tạo dataset với transform cho texture
gso_dataset = GSO_Dataset("path/to/dataset", transform=transforms.ToTensor())

# Tạo DataLoader
dataloader = DataLoader(gso_dataset, batch_size=4, shuffle=True)

# Kiểm tra dữ liệu
sample = next(iter(dataloader))
print(sample["mesh_path"])
print(sample["texture_path"])
