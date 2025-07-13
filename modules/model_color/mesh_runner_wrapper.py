import importlib.util
import sys
from pathlib import Path

# ✅ Thêm thư mục models của Point2Mesh vào sys.path TRƯỚC khi import mesh_runner
point2mesh_root = Path(__file__).resolve().parent.parent.parent / "Surface-Reconstruction-from-Point-Cloud-Point2Mesh"
point2mesh_models = point2mesh_root / "models"

if str(point2mesh_models) not in sys.path:
    sys.path.insert(0, str(point2mesh_models))  # 👈 Ưu tiên models đúng

# ✅ Import an toàn qua alias
mesh_runner_path = point2mesh_root / "mesh_runner.py"
spec = importlib.util.spec_from_file_location("mesh_runner_point2mesh", mesh_runner_path)
mesh_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mesh_runner)

# Public API
run_mesh = mesh_runner.run_mesh
