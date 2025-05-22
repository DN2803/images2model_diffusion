import numpy as np
import open3d as o3d

class PointCloudProcessing:
    def __init__(self, filepath):
        self.pcd = o3d.io.read_point_cloud(filepath)

    def filter_with_dbscan(self, eps=0.13, min_points=3):
        """
        Lọc nhiễu sử dụng thuật toán DBSCAN.
        """
        print("Filtering outliers using DBSCAN...")
        labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
        if labels.max() < 0:
            print("No clusters found.")
            return

        # Giữ lại cụm có số lượng điểm lớn nhất
        largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
        indices = np.where(labels == largest_cluster)[0]
        self.pcd = self.pcd.select_by_index(indices)
        print("Filtered point cloud.")

    def remove_outliers(self):
        """
        Lọc nhiễu bằng phương pháp thống kê (Statistical Outlier Removal).
        """
        print("Removing noise using Statistical Outlier Removal...")
        cl_stat, ind_stat = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.display_inlier_outlier(self.pcd, ind_stat)
        self.pcd = self.pcd.select_by_index(ind_stat)

    def display_inlier_outlier(self, pcd, ind_stat):
        """
        Hiển thị các điểm inlier và outlier.
        """
        inlier_cloud = pcd.select_by_index(ind_stat)
        outlier_cloud = pcd.select_by_index(ind_stat, invert=True)
        print("Displaying inliers and outliers...")
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="Inliers and Outliers")
    

    def xyz_to_ply_nocolor(self, output_ply_file):
        """
        Chuyển đổi đám mây điểm từ numpy array (XYZ) thành file .ply.
        """
        points = np.asarray(self.pcd.points)

        # Tạo đối tượng point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Ước lượng vector pháp tuyến
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        # Căn chỉnh hướng pháp tuyến
        pcd.orient_normals_consistent_tangent_plane(k=30)

        # Lưu ra file .ply có pháp tuyến
        o3d.io.write_point_cloud(output_ply_file, pcd, write_ascii=True) 



    def xyz_to_ply_color(self, output_ply_file):
        """
        Chuyển đổi đám mây điểm từ numpy array (XYZ) thành file .ply, giữ lại màu và pháp tuyến.
        """
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors) if self.pcd.has_colors() else None

        # Tạo đối tượng point cloud mới
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Giữ lại màu nếu có
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Ước lượng và căn chỉnh pháp tuyến
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)

        # Lưu ra file .ply với màu và pháp tuyến
        o3d.io.write_point_cloud(output_ply_file, pcd, write_ascii=True)


    def transfer_colors_from_pcd_to_mesh_file(self, mesh_path, output_colored_mesh_path=None):
        """
        Đọc mesh từ file và ánh xạ màu từ point cloud sang mesh dựa trên vị trí đỉnh.

        Parameters:
        - mesh_path (str): Đường dẫn đến file mesh (.obj, .ply, .stl,...)
        - output_colored_mesh_path (str, optional): Nếu có, lưu mesh có màu ra file.

        Returns:
        - mesh (TriangleMesh): Mesh đã được gán màu (có thể hiển thị hoặc lưu).
        """
        # Đọc mesh từ file
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.is_empty():
            print("Failed to load mesh.")
            return None

        if not self.pcd.has_colors():
            print("Point cloud has no color information.")
            return mesh

        print("Transferring colors from point cloud to mesh...")
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        mesh_colors = []

        for vertex in mesh.vertices:
            [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
            mesh_colors.append(self.pcd.colors[idx[0]])

        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        self.mesh = mesh  # lưu lại nếu cần dùng lại

        # Lưu nếu được yêu cầu
        if output_colored_mesh_path:
            o3d.io.write_triangle_mesh(output_colored_mesh_path, mesh, write_ascii=True)
            print(f"Saved colored mesh to {output_colored_mesh_path}")

        return mesh

    
    def save_ply_file(self, output_file):
        """
        Lưu point cloud hiện tại ra file .ply (không thêm hoặc tính toán gì thêm).
        """
        o3d.io.write_point_cloud(output_file, self.pcd, write_ascii=True)
        print(f"Saved point cloud to {output_file}")


# # Đọc và xử lý đám mây điểm từ file .ply
# ply_input_file = "redbui.ply"
# ply_output_file = "phatpointcloud.ply"

# pcd_processor = PointCloudProcessing(ply_input_file)

# # Lọc nhiễu bằng DBSCAN và SOR
# pcd_processor.filter_with_dbscan(eps=0.13, min_points=3)
# pcd_processor.remove_outliers()
# #pcd_processor.remove_outliers()



# # # Chuyển đổi lại đám mây điểm sau khi xử lý sang file .ply
# #pcd_processor.xyz_to_ply_nocolor(ply_output_file)
# pcd_processor.xyz_to_ply_color(ply_output_file)

# #pcd_processor.save_ply_file(ply_output_file)

# mesh_path = "ao vl.obj"  # hoặc .ply
# output_path = "colored_mesh_red.obj"

# pcd_processor.transfer_colors_from_pcd_to_mesh_file(mesh_path, output_colored_mesh_path=output_path)




