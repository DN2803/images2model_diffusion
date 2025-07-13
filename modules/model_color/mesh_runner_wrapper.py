import importlib.util
from pathlib import Path

# Đường dẫn tuyệt đối tới mesh_runner.py trong Point2Mesh
mesh_runner_path = Path(__file__).resolve().parent.parent.parent / "Surface-Reconstruction-from-Point-Cloud-Point2Mesh/mesh_runner.py"

# Load module với tên alias
spec = importlib.util.spec_from_file_location("mesh_runner_alias", mesh_runner_path)
mesh_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mesh_runner)

# Gán lại hàm cần dùng
run_mesh = mesh_runner.run_mesh
