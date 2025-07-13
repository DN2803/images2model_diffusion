from pathlib import Path
from glob import glob
import os
from utils.format_ply_file import PointCloudProcessing
from modules.model_color.mesh_runner_wrapper import run_mesh  # ✅ dùng wrapper an toàn

def mesh_generate(pcl: Path, output_dir: Path):
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

    ply_files = sorted(glob(os.path.join(output_dir, "mesh_*.ply")))
    return ply_files
