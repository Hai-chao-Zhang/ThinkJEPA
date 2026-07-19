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
'''
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

List of skeletal transforms in the HDF5 files.
'''

# left fingers
LEFT_LITTLE = ['leftLittleFingerMetacarpal', 'leftLittleFingerKnuckle', 'leftLittleFingerIntermediateBase',  'leftLittleFingerIntermediateTip', 'leftLittleFingerTip']
LEFT_RING = [ 'leftRingFingerMetacarpal', 'leftRingFingerKnuckle', 'leftRingFingerIntermediateBase', 'leftRingFingerIntermediateTip', 'leftRingFingerTip']
LEFT_MIDDLE = ['leftMiddleFingerMetacarpal', 'leftMiddleFingerKnuckle', 'leftMiddleFingerIntermediateBase', 'leftMiddleFingerIntermediateTip', 'leftMiddleFingerTip']
LEFT_INDEX = ['leftIndexFingerMetacarpal', 'leftIndexFingerKnuckle', 'leftIndexFingerIntermediateBase', 'leftIndexFingerIntermediateTip', 'leftIndexFingerTip']
LEFT_THUMB = ['leftThumbKnuckle', 'leftThumbIntermediateBase', 'leftThumbIntermediateTip', 'leftThumbTip',]
LEFT_FINGERS = LEFT_LITTLE + LEFT_RING + LEFT_MIDDLE + LEFT_INDEX + LEFT_THUMB

# right fingers
RIGHT_LITTLE = ['rightLittleFingerMetacarpal', 'rightLittleFingerKnuckle', 'rightLittleFingerIntermediateBase',  'rightLittleFingerIntermediateTip', 'rightLittleFingerTip']
RIGHT_RING = [ 'rightRingFingerMetacarpal', 'rightRingFingerKnuckle', 'rightRingFingerIntermediateBase', 'rightRingFingerIntermediateTip', 'rightRingFingerTip']
RIGHT_MIDDLE = ['rightMiddleFingerMetacarpal', 'rightMiddleFingerKnuckle', 'rightMiddleFingerIntermediateBase', 'rightMiddleFingerIntermediateTip', 'rightMiddleFingerTip']
RIGHT_INDEX = ['rightIndexFingerMetacarpal', 'rightIndexFingerKnuckle', 'rightIndexFingerIntermediateBase', 'rightIndexFingerIntermediateTip', 'rightIndexFingerTip']
RIGHT_THUMB = ['rightThumbKnuckle', 'rightThumbIntermediateBase', 'rightThumbIntermediateTip', 'rightThumbTip',]
RIGHT_FINGERS = RIGHT_LITTLE + RIGHT_RING + RIGHT_MIDDLE + RIGHT_INDEX + RIGHT_THUMB

# left arm
LEFT_ARM = ['leftShoulder', 'leftArm', 'leftForearm', 'leftHand']

# right arm
RIGHT_ARM = ['rightShoulder', 'rightArm', 'rightForearm', 'rightHand']

# spine
SPINE = ['hip', 'spine1', 'spine2', 'spine3', 'spine4', 'spine5', 'spine6', 'spine7']

# neck
NECK = ['neck1', 'neck2', 'neck3', 'neck4']

# wrists
WRISTS = ['leftHand', 'rightHand']

DEFAULT_TFS = set(LEFT_FINGERS + LEFT_ARM + RIGHT_FINGERS + RIGHT_ARM + SPINE + NECK)