import cv2
import numpy as np
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, labels):
        for t in self.transforms:
            image, labels = t(image, labels)
        return image, labels


class RandomCrop(object):
    def __init__(self, jitter=0.2):
        self.jitter = jitter

    def __call__(self, image, labels):
        oh, ow, oc = image.shape
        dw = int(ow * self.jitter)
        dh = int(oh * self.jitter)

        pleft = int(random.uniform(-dw, dw))
        pright = int(random.uniform(-dw, dw))
        ptop = int(random.uniform(-dh, dh))
        pbot = int(random.uniform(-dh, dh))

        cropped = self.crop_image(image, pleft, ptop, pright, pbot)
        crop_h, crop_w, _ = cropped.shape

        labels_out = labels.copy()
        shift = np.array([pleft, ptop, pleft, ptop])  # [m, 4]
        labels_out[:, 1:] = labels_out[:, 1:] - shift
        labels_out[:, 1] = labels_out[:, 1].clip(min=0, max=crop_w)
        labels_out[:, 2] = labels_out[:, 2].clip(min=0, max=crop_h)
        labels_out[:, 3] = labels_out[:, 3].clip(min=0, max=crop_w)
        labels_out[:, 4] = labels_out[:, 4].clip(min=0, max=crop_h)

        mask_w = ((labels_out[:, 3] - labels_out[:, 1]) / crop_w > 0.001)
        mask_h = ((labels_out[:, 4] - labels_out[:, 2]) / crop_h > 0.001)
        labels_out = labels_out[mask_w & mask_h]

        if len(labels_out) == 0:
            return image, labels

        return cropped, labels_out

    def crop_image(self, img, pleft, ptop, pright, pbot):
        oh, ow, oc = img.shape

        (xmin, left) = (0, -pleft) if pleft < 0 else (pleft, 0)
        (ymin, top) = (0, -ptop) if ptop < 0 else (ptop, 0)
        (xmax, right) = (ow, -pright) if pright < 0 else (ow - pright, 0)
        (ymax, bot) = (oh, -pbot) if pbot < 0 else (oh - pbot, 0)

        cropped_img = np.ascontiguousarray(img[ymin:ymax, xmin:xmax, :]).copy()
        pad_img = cv2.copyMakeBorder(cropped_img, top, bot, left, right, cv2.BORDER_REPLICATE)

        return pad_img


class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, labels):
        _, width, _ = image.shape
        if random.random() < self.prob:
            image = np.ascontiguousarray(image[:, ::-1])
            labels_cp = labels.copy()
            labels[:, 1::2] = width - labels_cp[:, 3::-2]

        return image, labels


class RandomHue(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, labels):
        if random.random() < self.prob:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.uniform(0.8, 1.2)
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return image, labels


class RandomSaturation(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, labels):
        if random.random() < self.prob:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.uniform(0.5, 1.5)
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return image, labels


class RandomBrightness(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, labels):
        if random.random() < self.prob:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.uniform(0.5, 1.5)
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return image, labels