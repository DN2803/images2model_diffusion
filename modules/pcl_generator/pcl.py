import open3d as o3d
from PIL import Image
import numpy as np
class PoitCloud:
    def __init__(self, data=None):
        self.data = data
    def show(self):
        # Hiển thị point cloud
        if self.data:
            o3d.visualization.draw_geometries([self.data])
    def save_obj(self, path):
        if self.data is None:
            print("No point cloud data to save.")
            return

        # Lấy các điểm và pháp tuyến từ point cloud
        points = np.asarray(self.data.points)
        normals = np.asarray(self.data.normals)

        with open(path, 'w') as file:
            # Lưu các điểm (vertices)
            for point in points:
                file.write(f"v {point[0]} {point[1]} {point[2]}\n")

            # Lưu các vector pháp tuyến (normals)
            for normal in normals:
                file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

            # Lưu các mặt (faces) nếu có
            # Trong trường hợp này, không có thông tin mặt, chỉ lưu điểm.
            # Bạn có thể thêm mặt ở đây nếu cần thiết (sử dụng chỉ số các điểm đã lưu)
            # faces = [(1, 2, 3), (4, 5, 6)]  # Ví dụ mặt giả định
            # for face in faces:
            #     file.write(f"f {face[0]} {face[1]} {face[2]}\n")
    def create_point_cloud_no_camera(self, depth_img: Image.Image, normal_img: Image.Image):
        # Chuyển đổi ảnh thành mảng numpy
        depth_array = np.array(depth_img)
        normal_array = np.array(normal_img)

        # Kích thước ảnh
        height, width = depth_array.shape

        # Tạo các mảng trống cho điểm và normal
        # create a camenea object's intrinsic parameters
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width, height, 1000, 1000, width / 2, height / 2)


        # Tạo point cloud từ ảnh depth và normal
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_array),
            intrinsic,
            depth_scale=1.0,
            depth_trunc=1000.0,
            stride=1,
            project_valid_depth_only=False,
            depth_origin=0.0
        )

        # save point cloud
        self.data = pcd 
        return pcd 
    

