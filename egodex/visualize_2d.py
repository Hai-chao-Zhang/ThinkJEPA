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

import argparse

from egodex.trajectory_dataset import SimpleDataset
from egodex.utils.trajectory_data_utils import project_transforms_to_camera_frame
from egodex.utils.draw_utils import draw_projected_trajectory, write_video_frames_to_mp4, map_hand_joints_to_colors
from egodex.utils.skeleton_tfs import (
    LEFT_FINGERS,
    LEFT_INDEX,
    LEFT_LITTLE,
    LEFT_MIDDLE,
    LEFT_RING,
    LEFT_THUMB,
    RIGHT_FINGERS,
    RIGHT_INDEX,
    RIGHT_LITTLE,
    RIGHT_MIDDLE,
    RIGHT_RING,
    RIGHT_THUMB,
)

QUERY_TFS = (
    RIGHT_FINGERS
    + ["rightHand", "rightForearm"]
    + LEFT_FINGERS
    + ["leftHand", "leftForearm"]
)


tf2idx = {k: i for i, k in enumerate(QUERY_TFS)}


def gather_finger_points(finger_tf_names, tfs_in_cam, right=True):
    hand_name = "rightHand"
    if not right:
        hand_name = "leftHand"

    finger_points = [
        tfs_in_cam[tf2idx[hand_name], :3, -1]
    ]  # grab 3D position from SE(3) pose
    for tfname in finger_tf_names:
        finger_points.append(tfs_in_cam[tf2idx[tfname], :3, -1])

    return finger_points



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


def main(args):
    # use simple pytorch dataset to load the data
    dataset = SimpleDataset(args.data_dir, query_tfs=QUERY_TFS)
    num_transitions = dataset.cumulative_len[args.num_episodes - 1]
    right_dict = {
        "index": RIGHT_INDEX,
        "thumb": RIGHT_THUMB,
        "middle": RIGHT_MIDDLE,
        "ring": RIGHT_RING,
        "little": RIGHT_LITTLE,
    }
    left_dict = {
        "index": LEFT_INDEX,
        "thumb": LEFT_THUMB,
        "middle": LEFT_MIDDLE,
        "ring": LEFT_RING,
        "little": LEFT_LITTLE,
    }

    out_imgs = []
    for i in range(num_transitions):
        tfs, cam_ext, cam_int, cam_img, _, _ = dataset[i]
        cam_img = (
            cam_img.detach().cpu().numpy().transpose(1, 2, 0)
        )  # C x H x W -> H x W x C

        # transform poses to camera frame
        tfs_in_cam = project_transforms_to_camera_frame(tfs, cam_ext)

        # draw hands
        render_hand_projection(right_dict, tfs_in_cam, cam_img, cam_int, tf2idx, right=True)
        render_hand_projection(left_dict, tfs_in_cam, cam_img, cam_int, tf2idx, right=False)

        out_imgs.append(cam_img)

    # write to video
    write_video_frames_to_mp4(out_imgs, args.output_mp4)
    print("Done. Video saved to: {}".format(args.output_mp4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="path to data directory")
    parser.add_argument(
        "--num_episodes", help="number of episodes to visualize", default=1
    )
    parser.add_argument(
        "--output_mp4", help="where to save the output video", default="output.mp4"
    )
    args = parser.parse_args()

    try:
        main(args)
    except ValueError as exp:
        print("Error:", exp)
