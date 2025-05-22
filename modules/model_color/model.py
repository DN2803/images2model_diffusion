import sys
import importlib.util
import os
from pathlib import Path

# 1. Thêm root repo của Point2Mesh vào sys.path
repo_root = os.path.abspath('./modules/model_color/Surface-Reconstruction-from-Point-Cloud-Point2Mesh')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# 2. Load main.py bằng importlib
main_path = os.path.join(repo_root, 'main.py')
spec = importlib.util.spec_from_file_location("mesh_module", main_path)
mesh_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mesh_module)

# 3. Gọi hàm run_mesh

def mesh_generate(raw_pcl: Path, output_dir: Path): 
    
    mesh_module.run_mesh(raw_pcl, output_dir)
