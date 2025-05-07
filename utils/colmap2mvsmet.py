#!/usr/bin/env python
"""
Copyright 2019, Jingyang Zhang and Yao Yao, HKUST. Model reading is provided by COLMAP.
Preprocess script.
"""

from __future__ import print_function

import collections
import struct
import numpy as np
import multiprocessing as mp
import os
import argparse
import shutil
import cv2
from pathlib import Path
from functools import partial
import PIL.Image

#============================ read_model.py ============================#
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def calc_score(pair, images, extrinsic, points3D, theta0, sigma1, sigma2):
    i, j = pair
    id_i = images[i+1].point3D_ids
    id_j = images[j+1].point3D_ids
    common_ids = [pid for pid in id_i if pid in id_j and pid != -1]

    ci = -extrinsic[i+1][:3, :3].T @ extrinsic[i+1][:3, 3]
    cj = -extrinsic[j+1][:3, :3].T @ extrinsic[j+1][:3, 3]

    score = 0
    for pid in common_ids:
        p = points3D[pid].xyz
        theta = np.degrees(np.arccos(np.dot(ci - p, cj - p) / (np.linalg.norm(ci - p) * np.linalg.norm(cj - p))))
        sigma = sigma1 if theta <= theta0 else sigma2
        score += np.exp(-((theta - theta0) ** 2) / (2 * sigma ** 2))
    return i, j, score

def colmap2mvsnet(colmap_folder_path: Path, images_path: Path, output_path: Path,
                  max_d: int = 0, interval_scale: float = 1.0,
                  theta0: float = 5.0, sigma1: float = 1.0, sigma2: float = 10.0,
                  test: bool = False, convert_format: bool = False):
    """
    Convert colmap to mvsnet format.
    :param colmap_folder_path: Path to the folder containing the COLMAP model.
    :param images_path: Path to the folder containing the original images.
    :param output_path: Path to the output folder.
    """
    # Chuyển sang đối tượng Path
    colmap_folder_path = Path(colmap_folder_path)
    images_path = Path(images_path)
    output_path = Path(output_path)
    
    # Đọc COLMAP model
    cameras, images, points3D = read_model(colmap_folder_path, ext='.txt')

    # Tạo thư mục đầu ra
    img_out_dir = output_path / "images"
    cam_out_dir = output_path / "cams"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    cam_out_dir.mkdir(parents=True, exist_ok=True)
    
    num_images = len(images)

    # Define intrinsic parameters
    param_type = {
        'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
        'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
        'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
        'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
        'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
        'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
        'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
        'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
        'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
        'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
        'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
    }


    # Intrinsics
    intrinsic = {}
    for camera_id, cam in cameras.items():
        params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
        if 'f' in param_type[cam.model]:
            params_dict['fx'] = params_dict['f']
            params_dict['fy'] = params_dict['f']
        K = np.array([
            [params_dict['fx'], 0, params_dict['cx']],
            [0, params_dict['fy'], params_dict['cy']],
            [0, 0, 1]
        ])
        intrinsic[camera_id] = K
    
    # Extrinsics
    extrinsic = {}
    for image_id, image in images.items():
        ext = np.eye(4)
        ext[:3, :3] = qvec2rotmat(image.qvec)
        ext[:3, 3] = image.tvec
        extrinsic[image_id] = ext

    # Depth range
    depth_ranges = {}
    for i in range(num_images):
        zs = []
        for p3d_id in images[i+1].point3D_ids:
            if p3d_id == -1: continue
            point = np.append(points3D[p3d_id].xyz, 1)
            z = (extrinsic[i+1] @ point)[2]
            zs.append(z)
        if not zs:
            depth_ranges[i+1] = (0, 1, 1, 1)
            continue
        zs_sorted = sorted(zs)
        depth_min = zs_sorted[int(len(zs) * .01)]
        depth_max = zs_sorted[int(len(zs) * .99)]

        if max_d == 0:
            K = intrinsic[images[i+1].camera_id]
            ext = extrinsic[i+1]
            R, t = ext[:3, :3], ext[:3, 3]
            p1 = np.array([K[0, 2], K[1, 2], 1])
            p2 = np.array([K[0, 2] + 1, K[1, 2], 1])
            P1 = np.linalg.inv(R) @ (np.linalg.inv(K) @ p1 * depth_min - t)
            P2 = np.linalg.inv(R) @ (np.linalg.inv(K) @ p2 * depth_min - t)
            depth_num = (1/depth_min - 1/depth_max) / (1/depth_min - 1/(depth_min + np.linalg.norm(P2 - P1)))
        else:
            depth_num = max_d
        depth_interval = (depth_max - depth_min) / (depth_num - 1) / interval_scale
        depth_ranges[i+1] = (depth_min, depth_interval, depth_num, depth_max)

    # View selection
    score_matrix = np.zeros((num_images, num_images))

    # def calc_score(pair):
    #     i, j = pair
    #     id_i = images[i+1].point3D_ids
    #     id_j = images[j+1].point3D_ids
    #     common_ids = [pid for pid in id_i if pid in id_j and pid != -1]

    #     ci = -extrinsic[i+1][:3, :3].T @ extrinsic[i+1][:3, 3]
    #     cj = -extrinsic[j+1][:3, :3].T @ extrinsic[j+1][:3, 3]

    #     score = 0
    #     for pid in common_ids:
    #         p = points3D[pid].xyz
    #         theta = np.degrees(np.arccos(np.dot(ci - p, cj - p) / (np.linalg.norm(ci - p) * np.linalg.norm(cj - p))))
    #         sigma = sigma1 if theta <= theta0 else sigma2
    #         score += np.exp(-((theta - theta0) ** 2) / (2 * sigma ** 2))
    #     return i, j, score
    pairs = [(i, j) for i in range(num_images) for j in range(i+1, num_images)]
    partial_calc_score = partial(calc_score, images=images, extrinsic=extrinsic,
                                points3D=points3D, theta0=theta0, sigma1=sigma1, sigma2=sigma2)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(partial_calc_score, pairs)

    for i, j, s in results:
        score_matrix[i, j] = s
        score_matrix[j, i] = s

    view_sel = []
    for i in range(num_images):
        sorted_ids = np.argsort(score_matrix[i])[::-1][:10]
        view_sel.append([(k, score_matrix[i, k]) for k in sorted_ids])

    # Write cams
    for i in range(num_images):
        cam_file = cam_out_dir / f"{i:08d}_cam.txt"
        with open(cam_file, 'w') as f:
            f.write('extrinsic\n')
            f.writelines(' '.join(map(str, extrinsic[i+1][j])) + '\n' for j in range(4))
            f.write('\nintrinsic\n')
            K = intrinsic[images[i+1].camera_id]
            f.writelines(' '.join(map(str, K[j])) + '\n' for j in range(3))
            f.write('\n%.6f %.6f %.6f %.6f\n' % depth_ranges[i+1])

    # Write pair.txt
    with open(output_path / 'pair.txt', 'w') as f:
        f.write(f"{num_images}\n")
        for i, neighbors in enumerate(view_sel):
            f.write(f"{i}\n{len(neighbors)} ")
            for j, s in neighbors:
                f.write(f"{j} {s:.6f} ")
            f.write("\n")

   # Rename or copy images
    for i in range(num_images):
        src_img = images_path / images[i+1].name
        dst_img = img_out_dir / f"{i:08d}.jpg"
        print(src_img)

        if convert_format:
            try:
                img = PIL.Image.open(src_img)
                # Chuyển sang chế độ RGB nếu không phải RGB (tránh lỗi với ảnh .png có alpha)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(dst_img, format='JPEG')
            except Exception as e:
                print(f"Lỗi khi chuyển đổi ảnh: {e}")
        else:
            shutil.copyfile(src_img, dst_img)

    if test:
        print("Test mode: No files written.")

#============================ read_model.py ============================#


if __name__ == '__main__':
    image_path = Path("D:/Thesis_2025/demo_exp/2025-04-19_10-58-56_7603/input_0.png")
    img = cv2.imread(str(image_path))  # Dùng str(...) ở đây
    if not image_path.exists():
        print("File không tồn tại!")
    if img is None:
        print("Không đọc được ảnh: kiểm tra lại đường dẫn hoặc tệp ảnh.")
#     parser = argparse.ArgumentParser(description='Convert colmap camera')

#     parser.add_argument('--dense_folder', type=str, help='Project dir.')

#     parser.add_argument('--max_d', type=int, default=0)
#     parser.add_argument('--interval_scale', type=float, default=1)

#     parser.add_argument('--theta0', type=float, default=5)
#     parser.add_argument('--sigma1', type=float, default=1)
#     parser.add_argument('--sigma2', type=float, default=10)

#     parser.add_argument('--test', action='store_true', default=False, help='If set, do not write to file.')
#     parser.add_argument('--convert_format', action='store_true', default=False, help='If set, convert image to jpg format.')

#     args = parser.parse_args()

#     image_dir = os.path.join(args.dense_folder, 'images')
#     model_dir = os.path.join(args.dense_folder, 'sparse')
#     cam_dir = os.path.join(args.dense_folder, 'cams')
#     renamed_dir = os.path.join(args.dense_folder, 'images')

#     cameras, images, points3d = read_model(model_dir, '.txt')
#     num_images = len(list(images.items()))

#     param_type = {
#         'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
#         'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
#         'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
#         'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
#         'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
#         'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
#         'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
#         'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
#         'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
#         'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
#         'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
#     }

#     # intrinsic
#     intrinsic = {}
#     for camera_id, cam in cameras.items():
#         params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
#         if 'f' in param_type[cam.model]:
#             params_dict['fx'] = params_dict['f']
#             params_dict['fy'] = params_dict['f']
#         i = np.array([
#             [params_dict['fx'], 0, params_dict['cx']],
#             [0, params_dict['fy'], params_dict['cy']],
#             [0, 0, 1]
#         ])
#         intrinsic[camera_id] = i
#     print('intrinsic[1]\n', intrinsic[1], end='\n\n')

#     # extrinsic
#     extrinsic = {}
#     for image_id, image in images.items():
#         e = np.zeros((4, 4))
#         e[:3, :3] = qvec2rotmat(image.qvec)
#         e[:3, 3] = image.tvec
#         e[3, 3] = 1
#         extrinsic[image_id] = e
#     print('extrinsic[1]\n', extrinsic[1], end='\n\n')

#     # depth range and interval
#     depth_ranges = {}
#     for i in range(num_images):
#         zs = []
#         for p3d_id in images[i+1].point3D_ids:
#             if p3d_id == -1:
#                 continue
#             transformed = np.matmul(extrinsic[i+1], [points3d[p3d_id].xyz[0], points3d[p3d_id].xyz[1], points3d[p3d_id].xyz[2], 1])
#             zs.append(np.asscalar(transformed[2]))
#         zs_sorted = sorted(zs)
#         # relaxed depth range
#         depth_min = zs_sorted[int(len(zs) * .01)]
#         depth_max = zs_sorted[int(len(zs) * .99)]
#         # determine depth number by inverse depth setting, see supplementary material
#         if args.max_d == 0:
#             image_int = intrinsic[images[i+1].camera_id]
#             image_ext = extrinsic[i+1]
#             image_r = image_ext[0:3, 0:3]
#             image_t = image_ext[0:3, 3]
#             p1 = [image_int[0, 2], image_int[1, 2], 1]
#             p2 = [image_int[0, 2] + 1, image_int[1, 2], 1]
#             P1 = np.matmul(np.linalg.inv(image_int), p1) * depth_min
#             P1 = np.matmul(np.linalg.inv(image_r), (P1 - image_t))
#             P2 = np.matmul(np.linalg.inv(image_int), p2) * depth_min
#             P2 = np.matmul(np.linalg.inv(image_r), (P2 - image_t))
#             depth_num = (1 / depth_min - 1 / depth_max) / (1 / depth_min - 1 / (depth_min + np.linalg.norm(P2 - P1)))
#         else:
#             depth_num = args.max_d
#         depth_interval = (depth_max - depth_min) / (depth_num - 1) / args.interval_scale
#         depth_ranges[i+1] = (depth_min, depth_interval, depth_num, depth_max)
#     print('depth_ranges[1]\n', depth_ranges[1], end='\n\n')

#     # view selection
#     score = np.zeros((len(images), len(images)))
#     queue = []
#     for i in range(len(images)):
#         for j in range(i + 1, len(images)):
#             queue.append((i, j))
#     def calc_score(inputs):
#         i, j = inputs
#         id_i = images[i+1].point3D_ids
#         id_j = images[j+1].point3D_ids
#         id_intersect = [it for it in id_i if it in id_j]
#         cam_center_i = -np.matmul(extrinsic[i+1][:3, :3].transpose(), extrinsic[i+1][:3, 3:4])[:, 0]
#         cam_center_j = -np.matmul(extrinsic[j+1][:3, :3].transpose(), extrinsic[j+1][:3, 3:4])[:, 0]
#         score = 0
#         for pid in id_intersect:
#             if pid == -1:
#                 continue
#             p = points3d[pid].xyz
#             theta = (180 / np.pi) * np.arccos(np.dot(cam_center_i - p, cam_center_j - p) / np.linalg.norm(cam_center_i - p) / np.linalg.norm(cam_center_j - p))
#             score += np.exp(-(theta - args.theta0) * (theta - args.theta0) / (2 * (args.sigma1 if theta <= args.theta0 else args.sigma2) ** 2))
#         return i, j, score
#     p = mp.Pool(processes=mp.cpu_count())
#     result = p.map(calc_score, queue)
#     for i, j, s in result:
#         score[i, j] = s
#         score[j, i] = s
#     view_sel = []
#     for i in range(len(images)):
#         sorted_score = np.argsort(score[i])[::-1]
#         view_sel.append([(k, score[i, k]) for k in sorted_score[:10]])
#     print('view_sel[0]\n', view_sel[0], end='\n\n')

#     # write
#     try:
#         os.makedirs(cam_dir)
#     except os.error:
#         print(cam_dir + ' already exist.')
#     for i in range(num_images):
#         with open(os.path.join(cam_dir, '%08d_cam.txt' % i), 'w') as f:
#             f.write('extrinsic\n')
#             for j in range(4):
#                 for k in range(4):
#                     f.write(str(extrinsic[i+1][j, k]) + ' ')
#                 f.write('\n')
#             f.write('\nintrinsic\n')
#             for j in range(3):
#                 for k in range(3):
#                     f.write(str(intrinsic[images[i+1].camera_id][j, k]) + ' ')
#                 f.write('\n')
#             f.write('\n%f %f %f %f\n' % (depth_ranges[i+1][0], depth_ranges[i+1][1], depth_ranges[i+1][2], depth_ranges[i+1][3]))
#     with open(os.path.join(args.dense_folder, 'pair.txt'), 'w') as f:
#         f.write('%d\n' % len(images))
#         for i, sorted_score in enumerate(view_sel):
#             f.write('%d\n%d ' % (i, len(sorted_score)))
#             for image_id, s in sorted_score:
#                 f.write('%d %f ' % (image_id, s))
#             f.write('\n')
#     for i in range(num_images):
#         if args.convert_format:
#             img = cv2.imread(os.path.join(image_dir, images[i+1].name))
#             cv2.imwrite(os.path.join(renamed_dir, '%08d.jpg' % i), img)
#         else:
#             shutil.copyfile(os.path.join(image_dir, images[i+1].name), os.path.join(renamed_dir, '%08d.jpg' % i))
    print("test mode: no files written.")
    colmap2mvsnet(
        colmap_folder_path=Path("D:/Thesis_2025/demo_exp/2025-04-19_10-58-56_7603/mvs/0"),
        images_path=Path("D:/Thesis_2025/demo_exp/2025-04-19_10-58-56_7603"),
        output_path=Path("D:/Thesis_2025/demo_exp/2025-04-19_10-58-56_7603/Dense"),
        
        test=True,
        convert_format=True,
    )