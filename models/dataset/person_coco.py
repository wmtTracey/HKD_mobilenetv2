from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
import random
from collections import defaultdict
from collections import OrderedDict

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# from models.dataset.JointsDataset import JointsDataset
from models.dataset.custom_coco import CustomCOCODataset
# from models.dataset.SimpleJointsDataset import SimpleJointsDataset
from utils.nms.nms import oks_nms
from utils import custom_cocoeval

# logger = logging.getLogger(__name__)


class PersonCOCODataset(CustomCOCODataset):
    '''
    keypoints={0: "nose",1: "left_eye",2: "right_eye",3: "left_ear",4: "right_ear",5: "left_shoulder",6: "right_shoulder",
        7: "left_elbow",8: "right_elbow",9: "left_wrist",10: "right_wrist",11: "left_hip",12: "right_hip",13: "left_knee",
        14: "right_knee",15: "left_ankle",16: "right_ankle"
    },
    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                     [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
    '''

    # for coco
    # joint_ids = list(range(0, 17))
    # num_joints = len(joint_ids)
    # flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
    #               [9, 10], [11, 12], [13, 14], [15, 16]]
    # EVAL_JOINTS = list(range(0, num_joints))
    # skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
    #             [6, 8], [7, 9], [8, 10], [0, 1], [0, 2], [1, 3], [2, 4]]

    # for person
    # joint_ids = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]
    # EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # num_joints = 11
    # flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    # skeleton = [(0, 1), (0, 2), (3, 4), (4, 6), (6, 8), (3, 5), (5, 7), (4, 10), (3, 9)]

    # for person 12
    # joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # num_joints = len(joint_ids)
    # EVAL_JOINTS = list(range(0, num_joints))
    # flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
    # skeleton = [[0, 2], [2, 4], [1, 3], [3, 5], [6, 8], [8, 10],
    #             [7, 9], [9, 11], [0, 1], [6, 7], [0, 6], [1, 7]]
    # scale_rate = 1.25

    # for person 12
    # joint_ids = [0, 1, 2, 5, 6]
    # num_joints = len(joint_ids)
    # EVAL_JOINTS = list(range(0, num_joints))
    # flip_pairs = [[1, 2], [3, 4]]
    # skeleton = [(0, 1), (0, 2), (3, 4)]
    # scale_rate = 1.25

    # joint_ids = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]
    # num_joints = len(joint_ids)
    # EVAL_JOINTS = list(range(0, num_joints))
    # flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    # skeleton = [(0, 1), (0, 2), (3, 4), (4, 6), (6, 8), (3, 5), (5, 7), (4, 10), (3, 9)]
    # scale_rate = 1.25

    def __init__(self, cfg, root, image_set, is_train, transform=None, shuffle=True):
        super(PersonCOCODataset, self).__init__(cfg, root, image_set, is_train, transform, shuffle)
