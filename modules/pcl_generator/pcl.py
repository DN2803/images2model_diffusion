import open3d as o3d
import numpy as np

from models.depth_estimate.run_mcc import run_MCC as mcc

from ultils.io import pcl2o3d
class PCL:
    def __init__(self, data=None):
        self.data = data

    def __fusion(self, pcds):
        #TODO: Implement fusion of point clouds

        # Thiết lập các tham số cho ICP
        threshold = 0.02  # Ngưỡng khoảng cách tối đa giữa các điểm
        trans_init = np.eye(4)  # Ma trận biến đổi ban đầu

        pcd = pcl2o3d.run(pcds[0])
        for i in range(1, len(pcds)):
            pcd_next = pcl2o3d.run(pcds[i])
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd_next, pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            pcd_next.transform(reg_p2p.transformation)
            pcd += pcd_next
        return pcd  
    @staticmethod
    def generate(self, color_image_paths ,depth_images_paths):

        pcds = []
        # TODO: Implement the conversion of depth image to point cloud
        for color_image_path, depth_image_path in zip(color_image_paths, depth_images_paths):
            pcd = mcc(image=color_image_path, point_cloud=depth_image_path)
            pcds.append(pcd)

        # TODO: Implement fusion of point clouds
        raw_pcd = self.__fusion(pcds)

        return raw_pcd
    