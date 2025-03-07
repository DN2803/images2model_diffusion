import open3d as o3d 

def run(pointclouds):
    points = pointclouds.points_packed().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if pointclouds.features_packed() is not None:
        colors = pointclouds.features_packed().cpu().numpy()
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd