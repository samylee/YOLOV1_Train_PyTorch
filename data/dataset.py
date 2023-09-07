import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset

from utils.utils import xywh2xyxy, xyxy2xywh


class VOCDataset(Dataset):
    def __init__(self, label_list, input_size=448, transform=None, is_train=True, S=7, B=2, C=20):
        super(VOCDataset, self).__init__()
        self.is_train = is_train
        self.input_size = input_size
        self.transform = transform
        self.S, self.B, self.C = S, B, C

        with open(label_list, 'r') as f:
            image_path_lines = f.readlines()

        self.images_path = []
        self.labels = []
        for image_path_line in image_path_lines:
            image_path = image_path_line.strip().split()[0]
            label_path = image_path.replace('JPEGImages', 'labels').replace('jpg', 'txt')
            if not os.path.exists(label_path):
                continue

            self.images_path.append(image_path)
            with open(label_path, 'r') as f:
                label_lines = f.readlines()

            labels_tmp = np.empty((len(label_lines), 5), dtype=np.float32)
            for i, label_line in enumerate(label_lines):
                labels_tmp[i] = [float(x) for x in label_line.strip().split()]
            self.labels.append(labels_tmp)

        assert len(self.images_path) == len(self.labels), 'images_path\'s length dont match labels\'s length'

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images_path[idx]), cv2.COLOR_BGR2RGB)
        labels = self.labels[idx]

        img_h, img_w, _ = image.shape

        if self.is_train and self.transform:
            labels = xywh2xyxy(labels, img_w, img_h)
            image, labels = self.transform(image, labels)
            img_h, img_w, _ = image.shape
            labels = xyxy2xywh(labels, img_w, img_h)

        # resize
        image_resize = cv2.resize(image, (self.input_size, self.input_size))

        # to torch
        image_resize = torch.from_numpy(image_resize.transpose((2, 0, 1))).float().div(255)
        targets = self.encode_labels(torch.from_numpy(labels))

        return image_resize, targets

    def __len__(self):
        return len(self.images_path)

    def encode_labels(self, labels):
        targets = torch.zeros((self.S * self.S * (self.C + 1 + 4)), dtype=torch.float32)
        for b in range(labels.size(0)):
            label = int(labels[b][0])
            x = labels[b][1]
            y = labels[b][2]
            w = labels[b][3]
            h = labels[b][4]

            # coordinate for cell
            col = int(x * self.S)
            row = int(y * self.S)

            x_cell = x * self.S - col
            y_cell = y * self.S - row

            obj_index = self.S * self.S * self.C + (row * self.S + col)
            # ignore second label in the same grid
            if targets[obj_index] == 1.0:
                continue

            targets[obj_index] = 1.0

            class_index = (row * self.S + col) * self.C
            targets[class_index + label] = 1.0

            box_index = self.S * self.S * (self.C + 1) + (row * self.S + col) * 4
            targets[box_index + 0] = x_cell
            targets[box_index + 1] = y_cell
            targets[box_index + 2] = w
            targets[box_index + 3] = h

        return targets