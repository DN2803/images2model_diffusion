#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Training preprocesses.
"""

from __future__ import print_function

import os
import time
import glob
import random
import math
import re
import sys

import cv2
import numpy as np
# import tensorflow as tf
import scipy.io
import urllib
# from tensorflow.python.lib.io import file_io
# FLAGS = tf.app.flags.FLAGS

class Config:
    pass

config = Config()
config.max_d = 128

def center_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0,1), keepdims=True)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)

def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam

def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    # print ('mask min max', min_depth, max_depth)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image

def load_cam(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = config.max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam
# def load_pfm(file):
#     color = None
#     width = None
#     height = None
#     scale = None
#     data_type = None
#     header = file.readline().decode('UTF-8').rstrip()

#     if header == 'PF':
#         color = True
#     elif header == 'Pf':
#         color = False
#     else:
#         raise Exception('Not a PFM file.')
#     dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
#     if dim_match:
#         width, height = map(int, dim_match.groups())
#     else:
#         raise Exception('Malformed PFM header.')
#     # scale = float(file.readline().rstrip())
#     scale = float((file.readline()).decode('UTF-8').rstrip())
#     if scale < 0: # little-endian
#         data_type = '<f'
#     else:
#         data_type = '>f' # big-endian
#     data_string = file.read()
#     data = np.fromstring(data_string, data_type)
#     shape = (height, width, 3) if color else (height, width)
#     data = np.reshape(data, shape)
#     data = cv2.flip(data, 0)
#     return data

# def write_pfm(file, image, scale=1):
#     file = open(file, mode='wb')
#     color = None

#     if image.dtype.name != 'float32':
#         raise Exception('Image dtype must be float32.')

#     image = np.flipud(image)

#     if len(image.shape) == 3 and image.shape[2] == 3: # color image
#         color = True
#     elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
#         color = False
#     else:
#         raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

#     file.write(('PF\n' if color else 'Pf\n').encode())
#     file.write(('%d %d\n' % (image.shape[1], image.shape[0])).encode())

#     endian = image.dtype.byteorder

#     if endian == '<' or endian == '=' and sys.byteorder == 'little':
#         scale = -scale

#     file.write('%f\n' % scale)

#     image_string = image.tostring()
#     file.write(image_string)

#     file.close()
def load_pfm(file):
    header = file.readline().decode('UTF-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_line = file.readline().decode('UTF-8').strip()
    dim_match = re.match(r'^(\d+)\s+(\d+)$', dim_line)
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('UTF-8').strip())
    if scale < 0:
        data_type = '<f'  # little-endian
    else:
        data_type = '>f'  # big-endian

    data_string = file.read()
    data = np.frombuffer(data_string, dtype=data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def write_pfm(file, image, scale=1):
    file = open(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(('PF\n' if color else 'Pf\n').encode())
    file.write(('%d %d\n' % (image.shape[1], image.shape[0])).encode())

    endian = image.dtype.byteorder
    if endian == '<' or (endian == '=' and sys.byteorder == 'little'):
        scale = -scale

    file.write(('%f\n' % scale).encode())

    image_string = image.tobytes()
    file.write(image_string)
    file.close()
