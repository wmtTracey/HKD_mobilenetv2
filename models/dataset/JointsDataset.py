from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from models.augment.augment_landm_crop import OrientationModel
from models.tools.transforms import get_affine_transform
from models.tools.transforms import affine_transform
from models.tools.transforms import fliplr_joints
from models.core import udp_offset
from utils import image_utils
logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        # self.num_joints = 0
        # self.pixel_std = 200
        # self.flip_pairs = []
        # fix BUG self.flip_pairs=self.flip_pairs
        # self.parent_ids = []
        self.is_train = is_train
        self.root = root
        self.image_set = image_set
        # self.data_format = cfg['DATASET']['DATA_FORMAT']
        self.scale_factor = cfg['DATASET']['SCALE_FACTOR']
        self.scale_rate = cfg['DATASET']['SCALE_RATE']
        self.rotation_factor = cfg['DATASET']['ROT_FACTOR']
        self.flip = cfg['DATASET']['FLIP']

        self.image_size = np.array(cfg['MODEL']['IMAGE_SIZE'])
        self.target_type = cfg['MODEL']['TARGET_TYPE']
        self.heatmap_size = np.array(cfg['MODEL']['EXTRA']['HEATMAP_SIZE'])
        self.sigma = cfg['MODEL']['EXTRA']['SIGMA']
        self.transform = transform
        self.db = []
        self.kpd = cfg['LOSS']['KPD']
        # self.random_crop = RandomLandmCrop(p=0.5, margin_rate=0.3)
        # self.random_crop = RandomLandmEdgeCrop(p=0.5, margin_rate=0.9)
        # self.random_crop = RandomLandmPaste(p=0.5, margin_rate=0.5,
        #                                     bg_dir=os.path.join(self.root, "bg_image/image"))
        self.orientation = OrientationModel(self.image_size[0], self.image_size[1])

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self, ):
        return len(self.db)

    def center_scale2rect(self, center, scale, pixel_std=200):
        w = pixel_std * scale[0]
        h = pixel_std * scale[1]
        x = center[0] - 0.5 * w
        y = center[1] - 0.5 * h
        rect = [x, y, w, h]
        return rect

    def vis_images(self, src_image, joints_3d, joints_3d_vis, center, scale):

        dst_rect = self.center_scale2rect(center, scale, pixel_std=self.pixel_std)
        src_image = src_image.copy()
        # c, s = self.adjust_center_scale(c, s, alpha=-20, beta=0.75)
        # c, s = self.adjust_center_scale(c, s, alpha=self.center_alpha, beta=self.scale_beta)

        joints = joints_3d[:, 0:2]
        # c = np.mean(joints,axis=0)
        src_image = image_utils.draw_points_text(src_image, [center], texts=["center"])
        src_image = image_utils.draw_key_point_in_image(src_image, [joints], self.skeleton, vis_id=True,thickness=1)
        src_image = image_utils.draw_image_rects(src_image, [dst_rect])
        # image_processing.cv_show_image("src_image", image_processing.resize_image(src_image, 800))
        image_utils.cv_show_image("src_image", src_image)
        # image_processing.cv_show_image("dst_image", image_processing.resize_image(dst_image, 800))

    def __getitem__(self, idx):
        # idx=0
        db_rec = copy.deepcopy(self.db[idx])
        image_file = db_rec['image']
        image_id = db_rec['image_id'] if 'image_id' in db_rec else ''
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        # if self.data_format == 'zip':
        #     from utils import zipreader
        #     data_numpy = zipreader.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # else:
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        center = db_rec['center']
        scale = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        rot = 0
        # input = data_numpy
        if self.is_train:
            # data_numpy, joints = self.random_crop(data_numpy, joints)
            # box = db_rec['box']
            # data_numpy, boxes, joints = self.orientation(data_numpy, np.asarray([box]), joints)
            # center, scale = self._box2cs(boxes[0])

            sf = self.scale_factor  # 0.25
            rf = self.rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.8 else 0
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        # print(idx,rot, image_file)
        # self.vis_images(data_numpy, joints, joints_vis, center, scale)
        if self.target_type == 'gaussian':
            trans = get_affine_transform(center, scale, rot, self.image_size)
            input = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                   flags=cv2.INTER_LINEAR)
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        elif self.target_type == 'offset':
            trans = udp_offset.get_warpmatrix(rot, center * 2.0, self.image_size - 1.0, scale)
            input = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                   flags=cv2.INTER_LINEAR)
            joints[:, 0:2] = udp_offset.rotate_points(joints[:, 0:2], rot, center, self.image_size, scale, False)
        else:
            Exception("Error:".format(self.target_type))
        # trans = get_affine_transform(center, scale, rot, self.image_size)
        # input = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
        #                        flags=cv2.INTER_LINEAR)
        # for i in range(self.num_joints):
        #     if joints_vis[i, 0] > 0.0:
        #         joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        # print(idx,rot, image_file)
        target, target_weight = self.generate_target(joints, joints_vis)
        # print(joints_vis,joints,target_weight)
        # self.vis_images(input, joints, joints_vis, center, scale)
        if self.transform:
            input = self.transform(input)
        # print(target_weight)
        # target_weight = np.multiply(target_weight, self.joints_weight)
        target_weight = torch.from_numpy(target_weight)
        target = torch.from_numpy(target)
        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': rot,
            'score': score,
            'image_id': image_id
        }

        return input, target, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std ** 2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
            ks = np.exp(-1.0 * (diff_norm2 ** 2) / ((0.2) ** 2 * 2.0 * area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)
            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # or br[0] < 0 or br[1] < 0 or ul[0] < 0 or ul[1] < 0:
                    # fix a bug: ul<0
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        elif self.target_type == 'offset':
            # self.heatmap_size: [48,64] [w,h]
            target = np.zeros((self.num_joints,
                               3,
                               self.heatmap_size[1] *
                               self.heatmap_size[0]),
                              dtype=np.float32)
            feat_width = self.heatmap_size[0]
            feat_height = self.heatmap_size[1]
            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)
            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.reshape((-1,))
            feat_y_int = feat_y_int.reshape((-1,))
            kps_pos_distance_x = self.kpd
            kps_pos_distance_y = self.kpd
            feat_stride = (self.image_size - 1.0) / (self.heatmap_size - 1.0)
            for joint_id in range(self.num_joints):
                mu_x = joints[joint_id][0] / feat_stride[0]
                mu_y = joints[joint_id][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds

                x_offset = (mu_x - feat_x_int) / kps_pos_distance_x
                y_offset = (mu_y - feat_y_int) / kps_pos_distance_y

                dis = x_offset ** 2 + y_offset ** 2
                keep_pos = np.where((dis <= 1) & (dis >= 0))[0]
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id, 0, keep_pos] = 1
                    target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                    target[joint_id, 2, keep_pos] = y_offset[keep_pos]
            target = target.reshape((self.num_joints * 3, self.heatmap_size[1], self.heatmap_size[0]))
        else:
            raise Exception("target_type:{}".format(self.target_type))
        return target, target_weight
