


import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
import cv2

def split_image(image, rows=3, cols=2):
        """
        Cắt ảnh thành nhiều phần theo số hàng và cột.

        Args:
            image (Image): ảnh được tạo sinh.
            rows (int): Số hàng cần cắt.
            cols (int): Số cột cần cắt.

        Returns:
            tuple: Tuple chứa các ảnh đã cắt dưới dạng đối tượng Image.Image.
        """
        img_height, img_width = image.shape[:2]
        cell_width, cell_height = img_width // cols, img_height // rows

        cropped_images = []
        for i in range(rows):
            for j in range(cols):
                left, upper = j * cell_width, i * cell_height
                right, lower = left + cell_width, upper + cell_height
                cropped_img = image[upper:lower, left:right]  # Cắt ảnh bằng slicing
                cropped_images.append(cropped_img)

        return tuple(cropped_images)

def create_point_cloud(depth_img: Image.Image, normal_img: Image.Image, intrinsic=None):
    # Chuyển đổi ảnh sang numpy array
    depth_array = np.array(depth_img, dtype=np.float32)
    normal_array = np.array(normal_img, dtype=np.float32)
    # Kích thước ảnh
    height, width = depth_array.shape

    # Nếu không có ma trận nội tại, dùng giá trị mặc định
    if intrinsic is None:
        intrinsic = np.array([[max(width, height), 0, width / 2],
                              [0, max(width, height), height / 2],
                              [0, 0, 1]])

    # Tách giá trị f_x, f_y, c_x, c_y
    f_x, f_y = intrinsic[0, 0], intrinsic[1, 1]
    c_x, c_y = intrinsic[0, 2], intrinsic[1, 2]

    # Tạo lưới tọa độ ảnh
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

    # Chuyển đổi pixel sang tọa độ 3D
    Z = (depth_array / 255.0) * 2 - 1
    X = (x_grid - c_x) * Z / f_x
    Y = (y_grid - c_y) * Z / f_y

    # # Loại bỏ điểm có depth = 0
    mask_depth = Z > 0

    # Loại bỏ điểm có normal không hợp lệ (NaN hoặc [0,0,0])
    if normal_array.ndim == 3 and normal_array.shape[-1] == 3:
        normals = (normal_array / 255.0) * 2 - 1  # Chuẩn hóa normal về [-1, 1]
        
        # Kiểm tra normal hợp lệ (không phải NaN, không phải [0,0,0])
        mask_normal = np.all(normals != 0, axis=-1) & ~np.isnan(normals).any(axis=-1)

        # Kết hợp cả hai điều kiện
        mask = mask_depth & mask_normal
        
        # Lọc dữ liệu
        X, Y, Z = X[mask], Y[mask], Z[mask]
        normals = normals[mask]
    else:
        raise ValueError("Normal image must có 3 kênh RGB")

    # Tạo Point Cloud
    points = np.stack((X, Y, Z), axis=-1)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    point_cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    return point_cloud


def get_extrinsic_matrix(pov_x, pov_y, camera_position):
    # Tạo ma trận xoay quanh trục Y (pov_x) và trục X (pov_y)
    R_y = R.from_euler('y', pov_x, degrees=True).as_matrix()
    R_x = R.from_euler('x', pov_y, degrees=True).as_matrix()
    
    # Kết hợp hai phép quay
    R_matrix = R_y @ R_x
    
    # Vị trí camera (Tịnh tiến)
    T = np.array(camera_position).reshape(3, 1)
    
    # Tạo ma trận ngoại tại [R | T]
    extrinsic = np.hstack((R_matrix, T))
    extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))  # Thêm hàng cuối [0,0,0,1]
    return extrinsic

def get_intrinsic_matrix(width, height, fov_x, fov_y=None):
    if fov_y is None:
        fov_y = fov_x  # Mặc định FOV_x = FOV_y nếu chỉ có một giá trị

    f_x = (width / 2) / np.tan(np.radians(fov_x / 2))
    f_y = (height / 2) / np.tan(np.radians(fov_y / 2))
    c_x, c_y = width / 2, height / 2

    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0,  0,  1]])
    return K
# Danh sách POV
pov_x = [30, 90, 150, 210, 270, 330]
pov_y = [20, -10, 20, -10, 20, -10]

# Danh sách ma trận ngoại tại
camera_positions = []
for x, y in zip(pov_x, pov_y):

    cam_pos = [np.sin(np.radians(x)) * 2, np.sin(np.radians(y)) * 2, np.cos(np.radians(x)) * 2]
    camera_positions.append(get_extrinsic_matrix(x, y, cam_pos))


# Danh sách ma trận nội tại
intrinsics = []
for x, y in zip(pov_x, pov_y):
    intrinsics.append(get_intrinsic_matrix(1, 1, x, y))


depth_image = cv2.imread("resized_depth_map.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
normal_image = cv2.imread("normals.png", cv2.IMREAD_UNCHANGED).astype(np.float32)

depth_maps = split_image(depth_image, rows=3, cols=2)
normal_maps = split_image(normal_image, rows=3, cols=2)

def pil_to_numpy(image):
    """Chuyển đổi PIL Image thành NumPy array."""
    return np.array(image).astype(np.float32)

# Chuyển đổi tất cả ảnh trong tuple thành NumPy arrays
depth_maps_np = tuple(pil_to_numpy(img) for img in depth_maps)
normal_maps_np = tuple(pil_to_numpy(img) for img in normal_maps)

# Truyền vào hàm create_point_cloud_no_camera

for i in range(len(depth_maps_np)):
    pcd = create_point_cloud(depth_maps_np[i], normal_maps_np[i], intrinsics[i])
    o3d.io.write_point_cloud(f"point_cloud_{i}.ply", pcd)


# Load các Point Cloud từ các góc nhìn
point_clouds = None
for i in range(6):

  point_clouds = [o3d.io.read_point_cloud(f'point_cloud_{i}.ply')]


# Chọn point cloud gốc (mốc để ghép tất cả lại)
merged_pcl = point_clouds[0]

def transform_point_cloud(pcd, camera_pose):
    """Chuyển Point Cloud về hệ tọa độ toàn cục."""
    points = np.asarray(pcd.points)  # Lấy tọa độ điểm (N, 3)
    
    # Thêm cột 1 vào để có dạng (N, 4)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))

    # Chuyển đổi hệ quy chiếu
    transformed_points = (camera_pose @ points_h.T).T[:, :3]

    # Gán lại cho Point Cloud
    pcd.points = o3d.utility.Vector3dVector(transformed_points)
    return pcd

def merge_point_clouds(pcd_list):
    """Hợp nhất nhiều Point Cloud với ICP."""
    merged_pcd = pcd_list[0]  # Chọn PCL đầu tiên làm gốc
    
    for pcd in pcd_list[1:]:
        # Thực hiện ICP để căn chỉnh pcd vào merged_pcd
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd, merged_pcd, max_correspondence_distance=0.02,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        
        # Áp dụng biến đổi ICP
        pcd.transform(reg_p2p.transformation)
        
        # Gộp vào PCL tổng
        merged_pcd += pcd
    
    # Làm mịn để giảm noise
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.005)
    
    return merged_pcd

# Chuyển PCL về hệ quy chiếu thế giới
for i in range(len(point_clouds)):
    point_clouds[i] = transform_point_cloud(point_clouds[i], camera_positions[i])

# Hợp nhất Point Cloud
final_pcd = merge_point_clouds(point_clouds)


def draw_camera(camera_extrinsics, size=0.1, color=[1, 0, 0]):
    """ Tạo một đối tượng 3D mô phỏng camera. """
    cameras = []
    for extrinsic in camera_extrinsics:
        # Tạo một mũi tên biểu diễn hướng nhìn của camera
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=size*0.2,
                                                       cone_radius=size*0.4,
                                                       cylinder_height=size,
                                                       cone_height=size*0.5)
        arrow.paint_uniform_color(color)  # Màu camera
        arrow.compute_vertex_normals()
        
        # Chuyển hướng của mũi tên theo hướng nhìn của camera
        arrow.transform(extrinsic)
        
        # Lưu camera vào danh sách
        cameras.append(arrow)
    
    return cameras

# Vẽ point cloud
final_pcd = o3d.io.read_point_cloud("merged_pcl3.ply")

# Vẽ các camera
camera_meshes = draw_camera(camera_positions, size=0.2)

# Hiển thị trong Open3D
o3d.visualization.draw_geometries([final_pcd] + camera_meshes)