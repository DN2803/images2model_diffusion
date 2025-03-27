import cv2
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter

# Load Normal Map và Depth Map
normal_map = cv2.imread("source/normal_image_0.png", cv2.IMREAD_COLOR).astype(np.float32) / 255.0
depth_map = cv2.imread("source/depth_image_0.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Chuyển Normal Map thành Gradient
nx, ny, nz = (normal_map[..., 0] * 2 - 1, 
              normal_map[..., 1] * 2 - 1, 
              normal_map[..., 2] * 2 - 1)

# Tạo Gradient X, Y từ Normal Map
grad_x = nx / (nz + 1e-5)
grad_y = ny / (nz + 1e-5)

# Tích hợp Gradient để tạo Height Map
height_map = np.zeros_like(depth_map)
for y in range(1, height_map.shape[0]):
    height_map[y, :] = height_map[y - 1, :] + grad_y[y, :]

for x in range(1, height_map.shape[1]):
    height_map[:, x] = height_map[:, x - 1] + grad_x[:, x]

# Kết hợp Depth Map + Height Map
final_height = gaussian_filter(depth_map * 0.5 + height_map * 0.5, sigma=2)

# Tạo Mesh từ Height Map
rows, cols = final_height.shape
vertices = []
faces = []

for i in range(rows):
    for j in range(cols):
        vertices.append([i, j, final_height[i, j]])

# Tạo faces từ lưới tam giác
for i in range(rows - 1):
    for j in range(cols - 1):
        v1 = i * cols + j
        v2 = v1 + 1
        v3 = v1 + cols
        v4 = v3 + 1
        faces.append([v1, v2, v3])
        faces.append([v2, v4, v3])

# Xuất Mesh
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.export("output_mesh.obj")

print("✅ Mesh đã được tạo và lưu thành output_mesh.obj")
scene = trimesh.Scene([mesh])
scene.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

# Load mesh
mesh = trimesh.load("output_mesh.obj")

# Lấy vertices và faces
vertices = np.array(mesh.vertices)
faces = np.array(mesh.faces)

# Tạo Figure 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Thêm các mặt vào plot
mesh_collection = Poly3DCollection(vertices[faces], alpha=0.5, edgecolor='k')
ax.add_collection3d(mesh_collection)

# Cài đặt góc nhìn
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

# Hiển thị
plt.show()
