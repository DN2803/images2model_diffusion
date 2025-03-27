import torch
import open3d as o3d
from pytorch3d.structures import Pointclouds

def run(pcd_torch):
    """
    Chuyển từ PyTorch3D PointClouds sang Open3D PointCloud
    """
    points = pcd_torch.points_padded()[0].cpu().numpy()  # (N, 3)
    
    if pcd_torch.features_padded() is not None:
        colors = pcd_torch.features_padded()[0].cpu().numpy()  # (N, 3)
    else:
        colors = None

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

    return pcd_o3d