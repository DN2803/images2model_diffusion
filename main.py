import open3d as o3d
import numpy as np
import cv2
import glob

# Danh sách đường dẫn đến ảnh
nocs_paths = sorted(glob.glob("source/normal_image_*.png"))
depth_paths = sorted(glob.glob("source/depth_image_*.png"))
mask_paths = sorted(glob.glob("source/depth_image_*.png"))  # Nếu có ảnh mask

# Các góc quay (Azimuth, Elevation)
azimuths = [30, 90, 150, 210, 270, 330]
elevations = [20, -10, 20, -10, 20, -10]

def compute_focal_length(H, W, fov_x_deg=47.1, fov_y_deg=47.1):
    """Tính toán fx, fy từ FOV và kích thước ảnh."""
    fx = W / (2 * np.tan(np.radians(fov_x_deg) / 2))
    fy = H / (2 * np.tan(np.radians(fov_y_deg) / 2))
    return fx, fy

def depth_to_nocs(depth_map, normal_map, mask=None):
    """Chuyển đổi Depth Map và NOCS Map thành Point Cloud"""
    H, W = depth_map.shape
    fx, fy = compute_focal_length(H, W)
    cx, cy = W // 2, H // 2  

    # Tạo lưới tọa độ pixel
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_map / 255 * 320  # Chuyển depth về mm
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    
    points_3D = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

    # Nếu có mask, chỉ giữ lại điểm hợp lệ
    if mask is not None:
        mask = (mask > 128) & (Z > 160)
        points_3D = points_3D[mask]
 

    # Chuẩn hóa tọa độ về [-1:1]
    min_vals = np.min(points_3D, axis=0)
    max_vals = np.max(points_3D, axis=0)
    nocs_map = 2 * (points_3D - min_vals) / (max_vals - min_vals) - 1

    nocs_map = nocs_map *320
    colors = normal_map.reshape(-1, 3) 

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(nocs_map)
    pcl.colors = o3d.utility.Vector3dVector(colors)
    return pcl

def rotation_matrix(azimuth_deg, elevation_deg):
    """Tạo ma trận xoay từ góc Azimuth và Elevation"""
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    R_y = np.array([[np.cos(azimuth), 0, np.sin(azimuth)], [0, 1, 0], [-np.sin(azimuth), 0, np.cos(azimuth)]])
    R_x = np.array([[1, 0, 0], [0, np.cos(elevation), -np.sin(elevation)], [0, np.sin(elevation), np.cos(elevation)]])
    return R_y @ R_x
def compute_fpfh_features(pcl, voxel_size=0.05):
    """Tính toán FPFH feature descriptors cho Point Cloud"""
    pcl_down = pcl.voxel_down_sample(voxel_size)
    pcl_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcl_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcl_down, fpfh
def apply_ransac(source_pcl, target_pcl, voxel_size=0.05):
    """Căn chỉnh Point Cloud bằng RANSAC"""
    source_down, source_fpfh = compute_fpfh_features(source_pcl, voxel_size)
    target_down, target_fpfh = compute_fpfh_features(target_pcl, voxel_size)

    distance_threshold = voxel_size * 1.5
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True, max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4, criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 1.0)
    )
    return ransac_result.transformation

def apply_icp(source_pcl, target_pcl, initial_transformation):
    """Căn chỉnh Point Cloud bằng ICP sau khi áp dụng RANSAC"""
    threshold = 0.05
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcl, target_pcl, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return icp_result.transformation

transformed_pcls = []
for i, (nocs_path, depth_path, mask_path) in enumerate(zip(nocs_paths, depth_paths, mask_paths)):
    nocs_map = cv2.imread(nocs_path).astype(np.float32) 
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
    
    pcl = depth_to_nocs(depth_map, nocs_map, mask)
   
    pcl.rotate(rotation_matrix(azimuths[i], elevations[i]), center=(0, 0, 0))
    transformed_pcls.append(pcl)

combined_pcl = transformed_pcls[0]
for i in range(1, len(transformed_pcls)):
    print(f"Aligning PCL {i} with RANSAC + ICP...")

    # Bước 1: Căn chỉnh ban đầu bằng RANSAC
    ransac_transformation = apply_ransac(transformed_pcls[i], combined_pcl)

    # Bước 2: Căn chỉnh tinh bằng ICP
    icp_transformation = apply_icp(transformed_pcls[i], combined_pcl, ransac_transformation)

    transformed_pcls[i].transform(icp_transformation)
    combined_pcl += transformed_pcls[i]

o3d.visualization.draw_geometries([combined_pcl])

# **TÍNH NORMAL CHO POINT CLOUD**
combined_pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
combined_pcl.orient_normals_consistent_tangent_plane(100)

# **POISSON SURFACE RECONSTRUCTION**
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(combined_pcl, depth=10)
vertices_to_remove = densities < np.percentile(densities, 5)
mesh.remove_vertices_by_mask(vertices_to_remove)

o3d.visualization.draw_geometries([mesh])