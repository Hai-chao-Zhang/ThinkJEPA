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
"""

import cv2
import numpy as np


def map_hand_joints_to_colors(tf_names):
    colors = []
    for tf in tf_names:
        if "little" in tf.lower():
            colors.append((0, 152, 191))  # light blue
        elif "ring" in tf.lower():
            colors.append((173, 255, 47))  # green yellow
        elif "middle" in tf.lower():
            colors.append((230, 245, 250))  # pale torquoise
        elif "index" in tf.lower():
            colors.append((255, 99, 71))  # tomato
        elif "thumb" in tf.lower():
            colors.append((238, 130, 238))  # violet
    return np.array(colors)


def write_video_frames_to_mp4(img_list, mp4_path, fps=30, fourcc=cv2.VideoWriter_fourcc(*"mp4v")):
    """
    img_list may be:
      - [ (H, W, 3), (H, W, 3), ... ]   a list of single frames
      - [ (T, H, W, 3), (T2, H, W, 3), ... ]  multiple video segments
      - a mix of both
    Everything is written into a single mp4 in order.
    """
    # Use one sample item to determine H and W
    sample = img_list[0]
    if sample.ndim == 4:
        _, H, W, _ = sample.shape
    elif sample.ndim == 3:
        H, W, _ = sample.shape
    else:
        raise ValueError(f"unsupported shape: {sample.shape}")

    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (W, H))

    for item in img_list:
        if item.ndim == 3:
            # Single frame
            frame = item
            frame = np.ascontiguousarray(frame)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        elif item.ndim == 4:
            # One video segment
            T = item.shape[0]
            for t in range(T):
                frame = item[t]
                frame = np.ascontiguousarray(frame)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        else:
            raise ValueError(f"unsupported item shape: {item.shape}")

    writer.release()


# def draw_projected_line(pointa, pointb, image, intrinsic, color=(0, 255, 0), thickness=5):
#     # project 3d points into 2d
#     pointa2, _ = cv2.projectPoints(
#         pointa, np.eye(3), np.zeros(3), intrinsic, distCoeffs=np.zeros(5)
#     )
#     pointb2, _ = cv2.projectPoints(
#         pointb, np.eye(3), np.zeros(3), intrinsic, distCoeffs=np.zeros(5)
#     )
#     pointa2 = pointa2.squeeze()
#     pointb2 = pointb2.squeeze()

#     # don't draw if the line is out of bounds
#     T, H, W, _ = image.shape
#     if (
#         ((pointb2[0] < 0).any() and (pointa2[0] > W).any())
#         or ((pointb2[1] < 0).any() and (pointa2[1] > H).any())
#         or ((pointa2[0] < 0).any() and (pointb2[0] > W).any())
#         or ((pointa2[1] < 0).any() and (pointb2[1] > H).any())
#     ):
#         return

#     # draws a line in-place
#     cv2.line(
#         image,
#         pointa2.astype(int),
#         pointb2.astype(int),
#         color=color,
#         thickness=thickness,
#     )
#     cv2.circle(image, (int(pointa2[0]), int(pointa2[1])), 15, color, -1)
#     cv2.circle(image, (int(pointb2[0]), int(pointb2[1])), 15, color, -1)


# def draw_projected_trajectory(points_list, image, intrinsic, color=(0, 255, 0)):
#     # draw a sequence of lines in-place
#     ptm = points_list[0]
#     for pt in points_list[1:]:
#         draw_projected_line(ptm, pt, image, intrinsic, color)
#         ptm = pt


def _project_point_to_image_plane(point3d, intrinsic):
    """
    point3d:
      - (3,) / (1,3): a single point
      - (T,3): one point per frame of a video
    intrinsic: (3,3)
    Returns:
      - single point: (2,)
      - multiple points: (T, 2)
    """
    pt = np.asarray(point3d, dtype=np.float32)
    K = np.asarray(intrinsic, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsic must be (3,3), got {K.shape}")

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    def _project_rows(rows):
        z = rows[:, 2:3]
        safe_z = np.where(np.abs(z) < 1e-6, np.nan, z)
        x_img = fx * (rows[:, 0:1] / safe_z) + cx
        y_img = fy * (rows[:, 1:2] / safe_z) + cy
        return np.concatenate([x_img, y_img], axis=1)

    if pt.ndim == 1:  # (3,)
        return _project_rows(pt.reshape(1, 3)).reshape(2)

    elif pt.ndim == 2:  # (T,3)
        return _project_rows(pt)

    else:
        raise ValueError(f"point3d with ndim={pt.ndim} not supported")


def draw_projected_line(pointa, pointb, image, intrinsic, color=(0, 255, 0), thickness=5, cirle=6):
    """
    image:
      - (H, W, 3): single image
      - (T, H, W, 3): video
    pointa / pointb:
      - matched to image: single frame -> (3,), video -> (T,3)
    """
    pa = _project_point_to_image_plane(pointa, intrinsic)
    pb = _project_point_to_image_plane(pointb, intrinsic)

    # Single image
    if image.ndim == 3:
        H, W = image.shape[:2]
        if pa.ndim != 1 or pb.ndim != 1:
            raise ValueError("single image expects single 2D point")

        if not (0 <= pa[0] < W and 0 <= pa[1] < H) and not (
            0 <= pb[0] < W and 0 <= pb[1] < H
        ):
            return

        pa_i = (int(pa[0]), int(pa[1]))
        pb_i = (int(pb[0]), int(pb[1]))
        cv2.line(image, pa_i, pb_i, color=color, thickness=thickness)
        cv2.circle(image, pa_i, cirle, color, -1)
        cv2.circle(image, pb_i, cirle, color, -1)
        return

    # Video
    elif image.ndim == 4:
        T, H, W = image.shape[0], image.shape[1], image.shape[2]

        # Here pa and pb are expected to be (T, 2)
        if pa.ndim == 1:
            # The 3D point is constant, so reuse it for every frame
            pa = np.repeat(pa[None, :], T, axis=0)
        if pb.ndim == 1:
            pb = np.repeat(pb[None, :], T, axis=0)

        for i in range(T):
            frame = np.ascontiguousarray(image[i])

            u1, v1 = pa[i]
            u2, v2 = pb[i]

            inside1 = (0 <= u1 < W) and (0 <= v1 < H)
            inside2 = (0 <= u2 < W) and (0 <= v2 < H)
            if not (inside1 or inside2):
                image[i] = frame
                continue

            p1 = (int(u1), int(v1))
            p2 = (int(u2), int(v2))
            cv2.line(frame, p1, p2, color=color, thickness=thickness)
            cv2.circle(frame, p1, 6, color, -1)
            cv2.circle(frame, p2, 6, color, -1)
            image[i] = frame
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def draw_projected_trajectory(points_list, image, intrinsic, color=(0, 255, 0), thickness=2, cirle=2):
    ptm = points_list[0]
    for pt in points_list[1:]:
        draw_projected_line(ptm, pt, image, intrinsic, color=color, thickness=thickness, cirle=2)
        ptm = pt
