# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-11-04 08:54:34
"""
import models.dataset as dataset
from models.dataset.coco import COCODataset
from models.dataset.person_coco import PersonCOCODataset



def load_dataset(cfg, root, image_set, is_train, transform, shuffle=True):
    if cfg['DATASET']['DATASET'].lower() == "coco":
        data = COCODataset(cfg, root, image_set, is_train, transform)
    elif cfg['DATASET']['DATASET'].lower() == "person_coco":
        # data = dataset.person_coco(cfg, root, image_set, is_train, transform, shuffle=shuffle)
        data = PersonCOCODataset(cfg, root, image_set, is_train, transform, shuffle=shuffle)
    elif cfg['DATASET']['DATASET'].lower() == "custom_coco":
        data = dataset.custom_coco(cfg, root, image_set, is_train, transform, shuffle=shuffle)
    elif cfg['DATASET']['DATASET'].lower() == "parser_coco":
        file = cfg['DATASET']['TRAIN_FILE'] if is_train else cfg['DATASET']['TEST_FILE']
        root = [root] if isinstance(root, str) else root
        image_set = [image_set] if isinstance(image_set, str) else image_set
        file = [file] if isinstance(file, str) else file
        # assert len(root) == len(image_set)
        # assert len(file) == len(image_set)
        data_list = []
        for r, s, f in zip(root, image_set, file):
            print("---" * 10)
            if not f:
                continue
            d = dataset.parser_coco(cfg, r, s, is_train, f, transform)
            print("load file :{},have images {}".format(f, len(d)))
            data_list.append(d)
        print("---" * 10)
        # data = dataset.ConcatDataset(data_list, resample=False, shuffle=shuffle)
        data = dataset.ConcatDataset(data_list, resample=is_train, shuffle=shuffle)
    elif cfg['DATASET']['DATASET'].lower() == "mpii":
        data = dataset.mpii(cfg, root, image_set, is_train, transform, shuffle=shuffle)
    elif cfg['DATASET']['DATASET'].lower() == "custom_mpii":
        data = dataset.custom_mpii(cfg, root, image_set, is_train, transform, shuffle=shuffle)
    elif cfg['DATASET']['DATASET'].lower() == "student_mpii":
        data = dataset.student_mpii(cfg, root, image_set, is_train, transform, shuffle=shuffle)
    else:
        raise Exception("Error: no dataset:{}".format(cfg.DATASET.DATASET))
    return data
