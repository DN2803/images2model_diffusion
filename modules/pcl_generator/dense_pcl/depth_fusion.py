#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Convert MVSNet output to Gipuma format for post-processing.
"""

from __future__ import print_function
from pathlib import Path

import argparse
import os
import shutil
from struct import *
import open3d as o3d
import numpy as np
import cv2
from utils.mvs.preprocess import *


def read_gipuma_dmb(path):
    with open(path, "rb") as fid:
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]

        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F").copy()
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_gipuma_dmb(path, image):
    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    channels = image_shape[2] if len(image_shape) == 3 else 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(path, "wb") as fid:
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)


def mvsnet_to_gipuma_dmb(in_path, out_path):
    image = load_pfm(in_path)
    write_gipuma_dmb(out_path, image)


def mvsnet_to_gipuma_cam(in_path, out_path):
    '''convert mvsnet camera to gipuma camera format'''

    cam = load_cam(open(in_path))

    extrinsic = cam[0:4][0:4][0]
    intrinsic = cam[0:4][0:4][1]
    intrinsic[3][0] = 0
    intrinsic[3][1] = 0
    intrinsic[3][2] = 0
    intrinsic[3][3] = 0
    projection_matrix = np.matmul(intrinsic, extrinsic)
    projection_matrix = projection_matrix[0:3][:]

    f = open(out_path, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()

    return

def fake_gipuma_normal(in_depth_path, out_normal_path):
    depth_image = read_gipuma_dmb(in_depth_path)
    shape = depth_image.shape

    normal_image = np.ones_like(depth_image).reshape((shape[0], shape[1], 1))
    normal_image = np.tile(normal_image, (1, 1, 3)) / 1.732050808

    mask = (depth_image > 0).astype(np.float32).reshape((shape[0], shape[1], 1))
    mask = np.tile(mask, (1, 1, 3))

    normal_image *= mask
    normal_image = np.float32(normal_image)

    write_gipuma_dmb(out_normal_path, normal_image)


def mvsnet_to_gipuma(dense_folder, gipuma_point_folder):
    dense_folder = Path(dense_folder)
    gipuma_point_folder = Path(gipuma_point_folder)

    image_folder = dense_folder / 'images'
    cam_folder = dense_folder / 'cams'
    depth_folder = dense_folder / 'depths_mvsnet'

    gipuma_cam_folder = gipuma_point_folder / 'cams'
    gipuma_image_folder = gipuma_point_folder / 'images'
    gipuma_point_folder.mkdir(parents=True, exist_ok=True)
    gipuma_cam_folder.mkdir(parents=True, exist_ok=True)
    gipuma_image_folder.mkdir(parents=True, exist_ok=True)

    for image_path in image_folder.iterdir():
        image_prefix = image_path.stem
        in_cam_file = cam_folder / f'{image_prefix}_cam.txt'
        out_cam_file = gipuma_cam_folder / f'{image_path.name}.P'
        mvsnet_to_gipuma_cam(in_cam_file, out_cam_file)

    for image_path in image_folder.iterdir():
        shutil.copy(str(image_path), str(gipuma_image_folder / image_path.name))

    gipuma_prefix = '2333__'
    for image_path in image_folder.iterdir():
        image_prefix = image_path.stem
        sub_depth_folder = gipuma_point_folder / f'{gipuma_prefix}{image_prefix}'
        sub_depth_folder.mkdir(parents=True, exist_ok=True)

        in_depth_pfm = depth_folder / f'{image_prefix}_prob_filtered.pfm'
        out_depth_dmb = sub_depth_folder / 'disp.dmb'
        fake_normal_dmb = sub_depth_folder / 'normals.dmb'

        mvsnet_to_gipuma_dmb(in_depth_pfm, out_depth_dmb)
        fake_gipuma_normal(out_depth_dmb, fake_normal_dmb)


def probability_filter(dense_folder, prob_threshold):
    dense_folder = Path(dense_folder)
    image_folder = dense_folder / 'images'
    depth_folder = dense_folder / 'depths_mvsnet'

    for image_path in image_folder.iterdir():
        prefix = image_path.stem
        init_depth_map_path = depth_folder / f'{prefix}_init.pfm'
        prob_map_path = depth_folder / f'{prefix}_prob.pfm'
        out_depth_map_path = depth_folder / f'{prefix}_prob_filtered.pfm'
        out_ply_path = depth_folder / f'{prefix}_filtered.ply'

        # Load the depth map and probability map
        depth_map = load_pfm(init_depth_map_path).copy()
        prob_map = load_pfm(prob_map_path).copy()

        # Filter the depth map based on probability
        depth_map[prob_map < prob_threshold] = 0

        # Write the filtered depth map back to file
        write_pfm(out_depth_map_path, depth_map)

        # Generate point cloud and save as PLY
        height, width = depth_map.shape
        valid_points = []

        for y in range(height):
            for x in range(width):
                if depth_map[y, x] > 0:  # Only consider valid depth values
                    # Convert (x, y) depth to 3D coordinates
                    depth = depth_map[y, x]
                    point = [x, y, depth]
                    valid_points.append(point)

        if valid_points:
            # Create Open3D point cloud from valid points
            points = np.array(valid_points)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)

            # Save point cloud to PLY file
            o3d.io.write_point_cloud(str(out_ply_path), point_cloud)
            print(f"PLY saved: {out_ply_path}")
        else:
            print(f"No valid points found for {prefix}. PLY not created.")


def depth_map_fusion(point_folder, fusibile_exe_path, disp_thresh, num_consistent):
    point_folder = Path(point_folder)
    cam_folder = point_folder / 'cams'
    image_folder = point_folder / 'images'
    depth_min = 0.001
    depth_max = 100000
    normal_thresh = 360

    cmd = (
        f"{fusibile_exe_path} "
        f"-input_folder {point_folder}/ "
        f"-p_folder {cam_folder}/ "
        f"-images_folder {image_folder}/ "
        f"--depth_min={depth_min} "
        f"--depth_max={depth_max} "
        f"--normal_thresh={normal_thresh} "
        f"--disp_thresh={disp_thresh} "
        f"--num_consistent={num_consistent}"
    )
    print(cmd)
    os.system(cmd)


def run_conversion(
    dense_folder,
    fusibile_exe_path='/content/gdrive/MyDrive/images2model_diffusion/fusibile/fusibile',
    prob_threshold=0.8,
    disp_threshold=0.01,
    num_consistent=1
):
    dense_folder = Path(dense_folder)
    point_folder = dense_folder / 'points_mvsnet'
    point_folder.mkdir(parents=True, exist_ok=True)

    print('filter depth map with probability map')
    probability_filter(dense_folder, prob_threshold)

    print('Convert mvsnet output to gipuma input')
    mvsnet_to_gipuma(dense_folder, point_folder)

    print('Run depth map fusion & filter')
    depth_map_fusion(point_folder, fusibile_exe_path, disp_threshold, num_consistent)




from plyfile import PlyData, PlyElement
# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scan_folder, out_folder, plyfilename):
    # the pair file
    pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)
    # TODO: hardcode size
    # used_mask = [np.zeros([296, 400], dtype=np.bool) for _ in range(nviews)]

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depths_mvsnet/{:0>8}_init.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'depths_mvsnet/{:0>8}_prob.pfm'.format(ref_view)))[0]
        photo_mask = confidence > 0.8

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depths_mvsnet/{:0>8}_init.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= 3
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))


        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)