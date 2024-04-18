import cv2
import numpy as np
import numbers
from numpy import random
from PIL import Image
from utils import file_utils, image_utils


def resize_image(image, landm, resize_width, resize_height):
    height, width, _ = np.shape(image)
    scale = [resize_width / width, resize_height / height]
    if landm.shape[1] > 2:
        scale += [0] * (landm.shape[1] - 2)
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    landm = np.asarray(landm * scale)
    return image, landm


class RandomLandmCrop(object):
    """ 实现随机裁剪"""

    def __init__(self, p=0.5, margin_rate=0.5):
        """
        :param p: 实现随机裁剪的概率
        :param margin_rate: 随机裁剪的幅度
        """
        self.p = p
        self.margin_rate = margin_rate

    def random_crop(self, img, landm):
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        if len(landm) > 0:
            xmin = max(0, np.min(landm[:, 0]))
            ymin = max(0, np.min(landm[:, 1]))
            xmax = min(w_img, np.max(landm[:, 0]))
            ymax = min(h_img, np.max(landm[:, 1]))
            max_box = [xmin, ymin, xmax, ymax]
        else:
            max_box = [int(w_img * 0.2), int(h_img * 0.2), int(w_img * 0.8), int(h_img * 0.8)]
        max_l_trans = max_box[0]
        max_u_trans = max_box[1]
        max_r_trans = w_img - max_box[2]
        max_d_trans = h_img - max_box[3]
        # 从边界随机向内裁剪
        crop_xmin = int(random.uniform(0, max_l_trans * self.margin_rate))
        crop_ymin = int(random.uniform(0, max_u_trans * self.margin_rate))
        crop_xmax = int(min(w_img, random.uniform(w_img - max_r_trans * self.margin_rate, w_img)))
        crop_ymax = int(min(h_img, random.uniform(h_img - max_d_trans * self.margin_rate, h_img)))
        img = img[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
        if len(landm) > 0:
            crop_bias = [crop_xmin, crop_ymin]
            if landm.shape[1] > 2:
                crop_bias += [0] * (landm.shape[1] - 2)
            landm = landm - crop_bias
            # landm[:, 0] = np.clip(landm[:, 0], 0, w_img)
            # landm[:, 1] = np.clip(landm[:, 1], 0, h_img)
        img, landm = resize_image(img, landm, w_img, h_img)
        return img, landm

    def __call__(self, img, landm):
        """
        Args:
            img (numpy Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            img, landm = self.random_crop(img, landm)
        return img, landm


class RandomLandmEdgeCrop(object):
    """ 实现随机边缘裁剪"""

    def __init__(self, p=0.5, margin_rate=0.5):
        """
        :param p: 实现随机裁剪的概率
        :param margin_rate: 随机裁剪的幅度
        """
        self.p = p
        self.margin_rate = margin_rate
        self.max_scale = 2 / 5

    def random_edge_crop(self, img, landm):
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        if len(landm) > 0:
            xmin = max(0, np.min(landm[:, 0]))
            ymin = max(0, np.min(landm[:, 1]))
            xmax = min(w_img, np.max(landm[:, 0]))
            ymax = min(h_img, np.max(landm[:, 1]))
            max_box = [xmin, ymin, xmax, ymax]
        else:
            max_box = [int(w_img * 0.2), int(h_img * 0.2), int(w_img * 0.8), int(h_img * 0.8)]
        max_l_trans = max_box[0]
        max_u_trans = max_box[1]
        max_r_trans = w_img - max_box[2]
        max_d_trans = h_img - max_box[3]
        if random.random() < 0.5:
            crop_xmin = int(random.uniform(max_l_trans * self.margin_rate, max_box[0]))
            crop_ymin = int(random.uniform(max_u_trans * self.margin_rate, max_box[1]))
            crop_xmax = int(min(w_img, random.uniform(w_img - max_r_trans * self.margin_rate, w_img)))
            crop_ymax = int(min(h_img, random.uniform(h_img - max_d_trans * self.margin_rate, h_img)))
        else:
            crop_xmin = int(random.uniform(0, max_l_trans * self.margin_rate))
            crop_ymin = int(random.uniform(0, max_u_trans * self.margin_rate))
            crop_xmax = int(
                min(w_img, random.uniform(max_box[2], max_box[2] + max_r_trans * (1 - self.margin_rate))))
            crop_ymax = int(
                min(h_img, random.uniform(max_box[3], max_box[3] + max_d_trans * (1 - self.margin_rate))))
        crop_xmin = min(crop_xmin, int(w_img * self.max_scale))
        crop_ymin = min(crop_ymin, int(h_img * self.max_scale))
        crop_xmax = max(crop_xmax, int(w_img * (1 - self.max_scale)))
        crop_ymax = max(crop_ymax, int(h_img * (1 - self.max_scale)))
        img = img[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
        if len(landm) > 0:
            crop_bias = [crop_xmin, crop_ymin]
            if landm.shape[1] > 2:
                crop_bias += [0] * (landm.shape[1] - 2)
            landm = landm - crop_bias
            # landm[:, 0] = np.clip(landm[:, 0], 0, w_img)
            # landm[:, 1] = np.clip(landm[:, 1], 0, h_img)
        img, landm = resize_image(img, landm, w_img, h_img)
        return img, landm

    def __call__(self, img, landm):
        """
        Args:
            img (numpy Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            img, landm = self.random_edge_crop(img, landm)
        return img, landm


class RandomLandmPaste(object):
    """ 实现随机边缘裁剪"""

    def __init__(self, p=0.5, margin_rate=0.5, bg_dir="bg_image/"):
        """
        :param p: 实现随机裁剪的概率
        :param margin_rate: 随机裁剪的幅度
        """
        self.p = p
        self.margin_rate = margin_rate
        self.max_scale = 2 / 5
        self.scale = [1.2, 1.2]
        self.bg_image_list = file_utils.read_files_lists(bg_dir, subname="")
        self.bg_nums = len(self.bg_image_list)
        self.random_edge_crop = RandomLandmEdgeCrop(p=1.0, margin_rate=0.9)
        self.random_crop = RandomLandmCrop(p=1.0, margin_rate=0.9)

    def random_read_bg_image(self, is_rgb=False, crop_rate=0.5):
        index = int(np.random.uniform(0, self.bg_nums))
        image_path = self.bg_image_list[index]
        # image_path = self.bg_image_list[0]
        image = cv2.imread(image_path)
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = image.shape
        xmin, ymin, xmax, ymax = self.extend_bboxes([0, 0, w_img, h_img], [0.8, 0.8])
        crop_xmin = int(random.uniform(0, xmin * crop_rate))
        crop_ymin = int(random.uniform(0, ymin * crop_rate))
        crop_xmax = int(min(w_img, random.uniform(w_img - xmax * crop_rate, w_img)))
        crop_ymax = int(min(h_img, random.uniform(h_img - ymax * crop_rate, h_img)))
        if random.random() < 0.5:
            image = image[:, ::-1, :]
        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
        return image

    @staticmethod
    def extend_bboxes(box, scale=[1.0, 1.0]):
        """
        :param box: [xmin, ymin, xmax, ymax]
        :param scale: [sx,sy]==>(W,H)
        :return:
        """
        sx = scale[0]
        sy = scale[1]
        xmin, ymin, xmax, ymax = box
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        ex_w = (xmax - xmin) * sx
        ex_h = (ymax - ymin) * sy
        ex_xmin = cx - 0.5 * ex_w
        ex_ymin = cy - 0.5 * ex_h
        ex_xmax = ex_xmin + ex_w
        ex_ymax = ex_ymin + ex_h
        ex_box = [ex_xmin, ex_ymin, ex_xmax, ex_ymax]
        return ex_box

    @staticmethod
    def cv_paste_image(im, mask, start_point=(0, 0)):
        """
        :param im:
        :param start_point:
        :param mask:
        :return:
        """
        xim, ymin = start_point
        shape = mask.shape  # h, w, d
        im[ymin:(ymin + shape[0]), xim:(xim + shape[1])] = mask
        return im

    @staticmethod
    def pil_paste_image(im, mask, landm, start_point=(0, 0)):
        """
        :param im:
        :param mask:
        :param start_point:
        :return:
        """
        out = Image.fromarray(im)
        mask = Image.fromarray(mask)
        out.paste(mask, start_point.copy())
        if landm.shape[1] > 2:
            start_point += [0] * (landm.shape[1] - 2)
        landm = landm + start_point
        return np.asarray(out), landm

    @staticmethod
    def get_random_point(w, h, max_box):
        dst_w = int(np.random.uniform(0, w - max_box[2]))
        dst_h = int(np.random.uniform(0, h - max_box[3]))
        print(w, h, dst_w, dst_h)
        return [dst_w, dst_h]

    @staticmethod
    def get_random_edge_point(w, h, max_box, margin_rate=0.9):
        xmin = int(np.random.uniform(0, w * (1 - margin_rate)))
        ymin = int(np.random.uniform(0, h * (1 - margin_rate)))
        xmax = int(np.random.uniform(w * margin_rate - max_box[2], w - max_box[2]))
        ymax = int(np.random.uniform(h * margin_rate - max_box[3], h - max_box[3]))
        points = [[xmin, int(np.random.uniform(0, h - max_box[3]))],
                  [int(np.random.uniform(0, w - max_box[2])), ymin],
                  [xmax, int(np.random.uniform(0, h - max_box[3]))],
                  [int(np.random.uniform(0, w - max_box[2])), ymax]]
        p = points[int(np.random.uniform(0, 4))]
        # p = points[2]
        return p

    def random_paste(self, img, landm, bg_image):
        if len(landm) > 0:
            h_bg, w_bg, _ = bg_image.shape
            h_img, w_img, _ = img.shape
            xmin = max(0, np.min(landm[:, 0]))
            ymin = max(0, np.min(landm[:, 1]))
            xmax = min(w_img, np.max(landm[:, 0]))
            ymax = min(h_img, np.max(landm[:, 1]))
            max_box = [xmin, ymin, xmax, ymax]
            # start_point = self.get_random_point(w_bg, h_bg, max_box)
            start_point = self.get_random_edge_point(w_bg, h_bg, max_box, margin_rate=self.margin_rate)
            img, landm = self.pil_paste_image(bg_image, img, landm, start_point=start_point)
            img, landm = resize_image(img, landm, w_img, h_img)
        return img, landm

    def __call__(self, img, landm):
        """
        Args:
            img (numpy Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            bg_image = self.random_read_bg_image()
            img, landm = self.random_edge_crop(img, landm)
            img, landm = self.random_paste(img, landm, bg_image)
        return img, landm


class RandomRot90():
    def __init__(self, p=0.5):
        """
        关键点随机旋转90
        """
        self.p = p

    def image_rot90(self, img, boxes, landm):
        h, w, _ = img.shape
        img = np.rot90(img, 1).copy()  # 顺时针90
        if len(landm) > 0:
            # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
            # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
            # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
            landm[:, [0, 1]] = landm[:, [1, 0]]
            landm[:, 1] = w - landm[:, 1]
        if len(boxes) > 0:
            boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
            boxes[:, [1, 3]] = w - boxes[:, [3, 1]]
        return img, boxes, landm

    def image_re_rot90(self, img, boxes, landm):
        h, w, _ = img.shape
        img = np.rot90(img, -1).copy()  # 逆时针90
        # img = np.rot90(img, 1).copy()  # 顺时针90
        if len(landm) > 0:
            # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
            # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
            # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
            landm[:, [0, 1]] = landm[:, [1, 0]]
            landm[:, 0] = h - landm[:, 0]
        if len(boxes) > 0:
            boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
            boxes[:, [0, 2]] = h - boxes[:, [2, 0]]
        return img, boxes, landm

    def __call__(self, img, boxes, landm):
        # 如果h > w，说明该图像是竖屏图片，否则是横屏图片
        if random.random() < self.p:
            if random.random() < 0.5:
                img, boxes, landm = self.image_re_rot90(img, boxes, landm)
            else:
                img, boxes, landm = self.image_rot90(img, boxes, landm)
        return img, boxes, landm


class PortraitMode():
    def __init__(self):
        """
        竖屏模式
        """
        self.rot = RandomRot90(p=1.0)

    def __call__(self, img, boxes, landm):
        h, w, _ = img.shape
        if h <= w:
            # 如果h < w，是横屏图片,需要旋转为竖屏
            img, boxes, landm = self.rot(img, boxes, landm)
        return img, boxes, landm


class LandscapeMode():
    def __init__(self):
        """
        横屏模式
        """
        self.rot = RandomRot90(p=1.0)

    def __call__(self, img, boxes, landm):
        h, w, _ = img.shape
        if h >= w:
            # 如果h > w，是竖屏图片,需要旋转为横屏
            img, boxes, landm = self.rot(img, boxes, landm)
        return img, boxes, landm


class OrientationModel():
    def __init__(self, width, height):
        """

        :param width:
        :param height:
        """
        self.aspect_ratio = height / width
        self.portrait = PortraitMode()  # 竖屏模式
        self.landscape = LandscapeMode()  # 横屏模式

    def __call__(self, img, boxes, landm):
        if self.aspect_ratio >= 1.0:
            img, boxes, landm = self.portrait(img, boxes, landm)  # 竖屏模式
        else:
            img, boxes, landm = self.landscape(img, boxes, landm)  # 横屏模式
        return img, boxes, landm


if __name__ == "__main__":
    from utils import image_utils

    input_size = [800, 800]
    # image_path = "test.jpg"
    image_path = "test1.jpg"

    src_landm = [[267, 282, 0],
                 [272, 227, 0],
                 [237, 307, 0],
                 [392, 167, 0]]
    src_landm = [[267, 282, 0], [272, 227, 0]]
    src_boxes = [[100, 130, 140, 240]]
    bg_dir = "/home/dm/data3/dataset/finger_keypoint/finger1/bg_image"
    # augment = RandomLandmEdgeCrop(p=1.0, margin_rate=0.5)
    # augment = RandomLandmPaste(p=1.0, margin_rate=0.99, bg_dir=bg_dir)
    augment = PortraitMode()
    # augment = LandscapeMode()
    # augment = OrientationModel(400, 300)
    # augment = RandomLandmCrop(p=1.0, margin_rate=0.5)
    src_image = image_utils.read_image(image_path, colorSpace="BGR")
    src_landm = np.asarray(src_landm)
    src_boxes = np.asarray(src_boxes)
    for i in range(1000):
        dst_image, boxes, points = augment(src_image.copy(), src_boxes.copy(), src_landm.copy())
        dst_image = image_utils.draw_image_boxes(dst_image.copy(), boxes)
        # dst_image = image_processing.draw_points_text(dst_image, [center], texts=["center"], drawType="simple")
        # points=np.asarray(points).reshape(-1,2)
        dst_image = image_utils.draw_landmark(dst_image, [points], color=(255, 0, 0), vis_id=True)
        print(boxes)
        cv2.imshow("image", dst_image)
        cv2.waitKey(0)
