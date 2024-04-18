# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import torch
import torch.optim as optim


def create_work_space(cfg, flag=""):
    stime = time.strftime('%Y-%m-%d-%H-%M')
    model_name = cfg['MODEL']['NAME']
    if "WIDTH_MULT" in cfg['MODEL']['EXTRA']:
        model_name = "{}_{}".format(model_name, cfg['MODEL']['EXTRA']['WIDTH_MULT'])
    filter = ""
    if "NUM_DECONV_FILTERS" in cfg['MODEL']['EXTRA']:
        filter = str(cfg['MODEL']['EXTRA']['NUM_DECONV_FILTERS'][0])
    name = [model_name, cfg['MODEL']['NUM_JOINTS'], cfg['MODEL']['IMAGE_SIZE'][0], cfg['MODEL']['IMAGE_SIZE'][1], filter,
            cfg['MODEL']['TARGET_TYPE'], cfg['DATASET']['ROT_FACTOR'], cfg['DATASET']['SCALE_RATE'], flag, cfg['DATASET']['DATASET'], stime]
    name = [str(n) for n in name if n]
    name = "_".join(name)
    work_dir = os.path.join(cfg['WORK_DIR'], name)
    return work_dir


def get_optimizer(cfg, model):
    optimizer = None
    if cfg['TRAIN']['OPTIMIZER'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg['TRAIN']['LR'],
            momentum=cfg['TRAIN']['MOMENTUM'],
            weight_decay=cfg['TRAIN']['WD'],
            nesterov=cfg['TRAIN']['NESTEROV']
        )
    elif cfg['TRAIN']['OPTIMIZER'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg['TRAIN']['LR']
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth.tar'))
