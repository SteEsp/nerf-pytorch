from load_blender import pose_spherical
import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

# x = x
# y = -y
# z = -z
T_world = np.eye(4)
T_world[1, 1] = -1
T_world[2, 2] = -1

# x = x
# z = -z
# y = -y
T_frame = np.eye(4)
T_frame[1, 1] = -1
T_frame[2, 2] = -1

def load_dtu_data(basedir, half_res=False, testskip=1):

    data_dir = basedir
    render_cameras_name = 'cameras_sphere.npz'

    camera_dict = np.load(os.path.join(data_dir, render_cameras_name))
    camera_dict = camera_dict
    images_lis = sorted(glob(os.path.join(data_dir, 'image/*.png')))
    n_images = len(images_lis)
    imgs = [cv.cvtColor(cv.imread(im_name), cv.COLOR_BGR2RGB) for im_name in images_lis]
    imgs = (np.array(imgs) / 255.).astype(np.float32) # [n_images, H, W, 3]
    imgs = imgs.astype(np.float32) 

    # world_mat is a projection matrix from world to image
    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    scale_mats_np = []

    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []

    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(intrinsics)
        pose = T_world @ pose @ T_frame
        pose_all.append(pose)

    poses = np.stack(pose_all).astype(np.float32) # [n_images, 4, 4]

    # Scaling (we assume the scene to render is inside a unit sphere at origin)
    vectors_norms = np.linalg.norm(poses[:, :3, 3], axis=1)
    scale = 1 / np.max(vectors_norms)
    # scale all vectors
    poses[:, :3, 3] *= scale

    H, W = imgs.shape[1], imgs.shape[2]
    focal = intrinsics_all[0][0, 0]

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv.resize(img, (W, H), interpolation=cv.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # render_poses = np.expand_dims(poses[14], axis=0) # [1, 4, 4]
    # i_split = [np.arange(len(imgs)), np.arange(0), np.array([14])]
    render_poses = poses[:10]
    i_split = [np.arange(len(imgs)), np.arange(0), np.arange(10)]

    return imgs, poses, render_poses, [H, W, focal], i_split