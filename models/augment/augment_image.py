import cv2
import numbers
import random
import PIL.Image as Image
import numpy as np
from torchvision import transforms


class ResizePadding(object):
    def __init__(self, size=[300, 300]):
        """
        等比例图像resize,保持原始图像内容比，避免失真,短边会0填充
        :param size:
        """
        self.size = tuple(size)

    def __call__(self, image, labels=None):
        is_pil = isinstance(image, Image.Image)
        image = np.asarray(image) if is_pil else image
        height, width, _ = image.shape
        scale = min([self.size[0] / width, self.size[1] / height])
        new_size = [int(width * scale), int(height * scale)]
        pad_w = self.size[0] - new_size[0]
        pad_h = self.size[1] - new_size[1]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        image = cv2.resize(image, (new_size[0], new_size[1]))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        image = Image.fromarray(image) if is_pil else image
        return image, labels


class RandomResize(object):
    """ random resize images"""

    def __init__(self, resize_range, interpolation=Image.BILINEAR):
        """
        :param resize_range: range size range
        :param interpolation:
        """
        self.interpolation = interpolation
        self.resize_range = resize_range

    def __call__(self, img):
        r = int(random.uniform(self.resize_range[0], self.resize_range[1]))
        size = (r, r)
        # print("RandomResize:{}".format(size))
        return transforms.functional.resize(img, size, self.interpolation)

    def __repr__(self):
        interpolation = transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolation)


class GaussianBlur(object):
    """Gaussian Blur for image"""

    def __init__(self):
        pass

    def __call__(self, image, ksize=(3, 3), sigmaX=0):
        is_pil = isinstance(image, Image.Image)
        image = np.asarray(image) if is_pil else image
        image = cv2.GaussianBlur(image, ksize, sigmaX)
        image = Image.fromarray(image) if is_pil else image
        return image


class RandomGaussianBlur(object):
    """Random Gaussian Blur for image"""

    def __init__(self, ksize=(1, 1, 1, 3, 3), sigmaX=0, p=0.5):
        """
        :param ksize: Gaussian kernel size. ksize.width and ksize.height can differ but they both must be
                      positive and odd. Or, they can be zero's and then they are computed from sigma.
        :param sigmaX:
        """
        self.ksize = ksize
        self.sigmaX = sigmaX
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            r = random.choice(self.ksize)
            is_pil = isinstance(image, Image.Image)
            image = np.asarray(image) if is_pil else image
            image = np.asarray(image)
            image = cv2.GaussianBlur(image, ksize=(r, r), sigmaX=self.sigmaX)
            image = Image.fromarray(image) if is_pil else image
        return image


class RandomMotionBlur(object):
    """
    Random Motion Blur for image运动模糊
    https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
    """

    def __init__(self, degree=5, angle=360, p=0.5):
        """
        :param degree: 运动模糊的程度，最小值为3
        :param angle: 运动模糊的角度,即模糊的方向，默认360度，即随机每个方向模糊
        :param p:
        """
        assert degree >= 3
        self.degree = degree
        self.angle = angle
        self.p = p

    def motion_blur(self, image, degree, angle):
        """
        :param image:
        :param degree: 运动模糊的程度,最小值为3
        :param angle: 运动模糊的角度,即模糊的方向
        :return:
        """
        # print("degree：{},angle：{}".format(degree, angle))
        angle = angle + 45  # np.diag对接矩阵是45度，所以需要补上45度
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        mat = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        kernel = np.diag(np.ones(degree))
        # print(degree, angle, mat, kernel)
        kernel = cv2.warpAffine(kernel, mat, (degree, degree))
        kernel = kernel / degree
        image = cv2.filter2D(image, -1, kernel)
        return image

    def __call__(self, image):
        if random.random() < self.p:
            is_pil = isinstance(image, Image.Image)
            image = np.asarray(image) if is_pil else image
            degree = int(random.uniform(3, self.degree + 1))
            angle = int(random.uniform(0, self.angle + 1))
            image = self.motion_blur(image, degree, angle)
            image = Image.fromarray(image) if is_pil else image
        return image


class RandomRotation(object):
    """
    旋转任意角度
    """

    def __init__(self, degrees=5, p=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            is_pil = isinstance(image, Image.Image)
            image = np.asarray(image) if is_pil else image
            angle = random.uniform(self.degrees[0], self.degrees[1])
            h, w, _ = image.shape
            center = (w / 2., h / 2.)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_mat, dsize=(w, h), borderMode=cv2.BORDER_CONSTANT)
            image = Image.fromarray(image) if is_pil else image
        return image


class RandomColorJitter(object):
    def __init__(self, p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1):
        """
        :param p:
        :param brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        :param contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        :param saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        :param hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.(色调建议设置0.1，避免颜色变化过大)
        """
        # from torchvision import transforms
        self.p = p
        self.random_choice = transforms.RandomChoice([
            transforms.ColorJitter(brightness=brightness),
            transforms.ColorJitter(contrast=contrast),
            transforms.ColorJitter(saturation=saturation),
            transforms.ColorJitter(hue=hue),
        ])
        self.color_transforms = transforms.ColorJitter(brightness=brightness,
                                                       contrast=contrast,
                                                       saturation=saturation,
                                                       hue=hue)

    def __call__(self, img):
        if random.random() < self.p:
            # img = self.random_choice(img)
            img = self.color_transforms(img)
        return img


class SwapChannels(object):
    """交换图像颜色通道的顺序"""

    def __init__(self, swaps=[], p=1.0):
        """
        由于输入可能是RGB或者BGR格式的图像，随机交换通道顺序可以避免图像通道顺序的影响
        :param swaps:指定交换的颜色通道顺序，如[2,1,0]
                     如果swaps=[]或None，表示随机交换顺序
        :param p:概率
        """
        self.p = p
        self.swap_list = []
        if not swaps:
            self.swap_list = [[0, 1, 2], [2, 1, 0]]
        else:
            self.swap_list = [swaps]
        self.swap_index = np.arange(len(self.swap_list))

    def __call__(self, image):
        if random.random() < self.p:
            is_pil = isinstance(image, Image.Image)
            image = np.asarray(image) if is_pil else image
            index = np.random.choice(self.swap_index)
            swap = self.swap_list[index]
            image = image[:, :, swap]
            image = Image.fromarray(image) if is_pil else image
        return image


class Crop(object):
    def __init__(self, crop_box):
        """
        :param crop_box: 裁剪类型
        """
        self.crop_box = crop_box

    def __call__(self, image):
        is_pil = isinstance(image, Image.Image)
        image = np.asarray(image) if is_pil else image
        h, w, _ = image.shape
        if isinstance(self.crop_box, list):
            xmin, ymin, xmax, ymax = self.crop_box
        elif self.crop_box == "body":
            xmin, ymin, xmax, ymax = (0, 0, w, int(h / 2))
        else:
            xmin, ymin, xmax, ymax = (0, 0, w, h)
        image = image[ymin:ymax, xmin:xmax]
        image = Image.fromarray(image) if is_pil else image
        return image


def demo_for_augment():
    from utils import image_utils
    input_size = [192, 256]
    image_path = "test.jpg"
    rgb_mean = [0., 0., 0.]
    rgb_std = [1., 1.0, 1.0]
    image = image_utils.read_image(image_path)
    augment = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([int(128 * input_size[1] / 112), int(128 * input_size[0] / 112)]),
        # RandomMotionBlur(),
        RandomGaussianBlur(),
        transforms.RandomCrop([input_size[1], input_size[0]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std),
    ])
    for i in range(1000):
        dst_image = augment(image.copy())
        dst_image = np.array(dst_image, dtype=np.float32)
        dst_image = dst_image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        print("{},dst_image.shape:{}".format(i, dst_image.shape))
        image_utils.cv_show_image("image", dst_image)
        print("===" * 10)


def demo_for_image():
    from utils import image_utils
    image_path = "test.jpg"
    image = image_utils.read_image(image_path)
    image = image_utils.resize_image(image, 256, 192)
    image_utils.cv_show_image("image", image, waitKey=10)
    augment = RandomMotionBlur()
    for angle in range(360):
        degree = 3
        dst_image = augment.motion_blur(image.copy(), degree, angle)
        print("{},{}，dst_image.shape:{}".format(degree, angle, dst_image.shape))
        image_utils.cv_show_image("dst_image", dst_image)
        print("===" * 10)


if __name__ == "__main__":
    demo_for_augment()
    # demo_for_image()
