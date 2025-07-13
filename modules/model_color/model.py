import sys
import os
from pathlib import Path
from glob import glob

from utils.format_ply_file import PointCloudProcessing

repo_root = Path('./Surface-Reconstruction-from-Point-Cloud-Point2Mesh')
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# 3. Gọi hàm run_mesh
from mesh_runner import run_mesh

def mesh_generate(pcl: Path, output_dir: Path):
    # Gọi thuật toán Point2Mesh để sinh ra nhiều mesh intermediate
    pcd_processor = PointCloudProcessing(pcl)
    no_color_ply = pcl.stem + '_nocolor.ply'
    pcd_processor.xyz_to_ply_nocolor(no_color_ply)
    run_mesh(
        input_pcl=no_color_ply,
        output_path=output_dir,
        args_list=[
        '--iterations', '3000',
        '--save-path', str(output_dir),
        '--input-pc', str(no_color_ply),
        ]
    )

    # Lấy danh sách mesh theo thứ tự thời gian (hoặc theo tên nếu đặt đúng)
    ply_files = sorted(glob(os.path.join(output_dir, "mesh_*.ply")))
    return ply_files