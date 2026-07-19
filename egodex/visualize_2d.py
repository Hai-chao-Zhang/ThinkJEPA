# ThinkJEPA
# Copyright (c) 2026 Northeastern University, Haichao Zhang, et al.
# This file is part of the ThinkJEPA release associated with:
#
# @article{zhang2026thinkjepa,
#   title={ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model},
#   author={Zhang, Haichao and Li, Yijiang and He, Shwai and Nagarajan, Tushar and Chen, Mingfei and Lu, Jianglin and Li, Ang and Fu, Yun},
#   journal={arXiv preprint arXiv:2603.22281},
#   year={2026}
# }
#
# See LICENSE and NOTICE for release terms.
#
"""
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Script for reprojecting 3D skeletal annotations into the 2D video.
Note that there may be some perspective error in the reprojection.
"""

from egodex.utils.draw_utils import draw_projected_trajectory, map_hand_joints_to_colors
def gather_finger_points(finger_tf_names, tfs_in_cam, tf2idx, right=True):
    """
    tfs_in_cam:
        - (N, 4, 4): single frame
        - (T, N, 4, 4): multiple frames
    Returns:
        - single frame: list[(3,)], e.g. 5 points
        - multiple frames: list[(T, 3)], e.g. 5 points each with T frames
    """
    if tfs_in_cam.ndim == 3:
        # Single frame (N, 4, 4)
        hand_name = "rightHand" if right else "leftHand"
        pts = [tfs_in_cam[tf2idx[hand_name], :3, 3]]
        for name in finger_tf_names:
            pts.append(tfs_in_cam[tf2idx[name], :3, 3])
        return pts

    # Video (T, N, 4, 4)
    hand_name = "rightHand" if right else "leftHand"
    pts = [tfs_in_cam[:, tf2idx[hand_name], :3, 3]]  # (T,3)
    for name in finger_tf_names:
        pts.append(tfs_in_cam[:, tf2idx[name], :3, 3])  # (T,3)
    return pts


def render_hand_projection(hand_dict, tfs_in_cam, cam_img, cam_int, tf2idx, right=True):
    """
    hand_dict: right_dict / left_dict
    tfs_in_cam:
        - (N, 4, 4)
        - (T, N, 4, 4)
    cam_img:
        - (H, W, 3)
        - (T, H, W, 3)
    cam_int: (3,3)
    """
    # 1) Draw the five fingers
    for finger in ["little", "ring", "middle", "index", "thumb"]:
        pts_list = gather_finger_points(hand_dict[finger], tfs_in_cam, tf2idx, right)
        draw_projected_trajectory(
            pts_list,
            cam_img,
            cam_int,
            color=map_hand_joints_to_colors([finger])[0].tolist(),
        )

    # 2) Draw the forearm
    if tfs_in_cam.ndim == 3:
        # Single frame
        if right:
            forearm_pts = [
                tfs_in_cam[tf2idx["rightForearm"], :3, 3],
                tfs_in_cam[tf2idx["rightHand"], :3, 3],
            ]
        else:
            forearm_pts = [
                tfs_in_cam[tf2idx["leftForearm"], :3, 3],
                tfs_in_cam[tf2idx["leftHand"], :3, 3],
            ]
    else:
        # Video
        if right:
            forearm_pts = [
                tfs_in_cam[:, tf2idx["rightForearm"], :3, 3],
                tfs_in_cam[:, tf2idx["rightHand"], :3, 3],
            ]
        else:
            forearm_pts = [
                tfs_in_cam[:, tf2idx["leftForearm"], :3, 3],
                tfs_in_cam[:, tf2idx["leftHand"], :3, 3],
            ]

    draw_projected_trajectory(
        forearm_pts,
        cam_img,
        cam_int,
        color=map_hand_joints_to_colors(["middle"])[0].tolist(),
    )
