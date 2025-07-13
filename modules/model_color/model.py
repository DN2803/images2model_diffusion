from pathlib import Path
from glob import glob
import os

from utils.format_ply_file import PointCloudProcessing
from modules.model_color.mesh_runner_wrapper import run_mesh  # Wrapper tránh lỗi xung đột `models`

def mesh_generate(pcl: Path, output_dir: Path):
    # B1: Tiền xử lý Point Cloud (chuyển XYZ → PLY không màu)
    pcd_processor = PointCloudProcessing(pcl)
    no_color_ply_path = output_dir / (pcl.stem + '_nocolor.ply')
    pcd_processor.xyz_to_ply_nocolor(no_color_ply_path)

    # B2: Gọi Point2Mesh để sinh mô hình 3D
    run_mesh(
        input_pcl=no_color_ply_path,
        output_path=output_dir,
        args_list=[
            '--iterations', '3000',
            '--save-path', str(output_dir),
            '--input-pc', str(no_color_ply_path),
        ]
    )

    # B3: Lấy danh sách file .obj được sinh ra (ví dụ: recon_iter_*.obj)
    result_dir = os.path.join(output_dir, "checkpoints", "result")
    obj_files = sorted(glob(os.path.join(result_dir, "recon_iter_*.obj")))

    # Thêm last_recon.obj nếu cần
    last_recon = os.path.join(result_dir, "last_recon.obj")
    if os.path.exists(last_recon):
        obj_files.append(last_recon)

    return obj_files
