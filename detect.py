import torch
import torch.nn as nn
import cv2
import numpy as np

from models.YOLOV1 import YOLOV1
from utils.utils import nms


def postprocess(output, thresh, S, B, C, img_w, img_h):
    # to cpu numpy
    predictions = output.squeeze(0).data.cpu().numpy()

    # detection results
    # [xmin, ymin, xmax, ymax, score, class_id]
    results = np.empty((0, 4 + 1 + 1), dtype=np.float32)

    probs_tmp = np.empty((C), dtype=np.float32)
    boxes_tmp = np.empty((4), dtype=np.float32)
    for i in range(S * S):
        row = i // S
        col = i % S

        # get obj
        prob_index = S * S * C + i * B
        obj1_prob = predictions[prob_index]
        obj2_prob = predictions[prob_index + 1]
        obj_prob_max = obj1_prob if obj1_prob > obj2_prob else obj2_prob
        obj_prob_max_index = 0 if obj1_prob > obj2_prob else 1

        # get class
        class_index = i * C
        for j in range(C):
            class_prob = obj_prob_max * predictions[class_index + j]
            probs_tmp[j] = class_prob if class_prob > thresh else 0

        if probs_tmp.max() > thresh:
            # get network boxes
            box_index = S * S * (C + B) + (i * B + obj_prob_max_index) * 4
            boxes_tmp[0] = (predictions[box_index + 0] + col) / S
            boxes_tmp[1] = (predictions[box_index + 1] + row) / S
            boxes_tmp[2] = pow(predictions[box_index + 2], 2)
            boxes_tmp[3] = pow(predictions[box_index + 3], 2)

            # get real boxes
            xmin = (boxes_tmp[0] - boxes_tmp[2] / 2.) * img_w
            ymin = (boxes_tmp[1] - boxes_tmp[3] / 2.) * img_h
            xmax = (boxes_tmp[0] + boxes_tmp[2] / 2.) * img_w
            ymax = (boxes_tmp[1] + boxes_tmp[3] / 2.) * img_h

            # limit rect
            xmin = xmin if xmin > 0 else 0
            ymin = ymin if ymin > 0 else 0
            xmax = xmax if xmax < img_w else img_w - 1
            ymax = ymax if ymax < img_h else img_h - 1

            values = [xmin, ymin, xmax, ymax, probs_tmp.max(), probs_tmp.argmax()]
            row_values = np.expand_dims(np.array(values), axis=0)
            results = np.append(results, row_values, axis=0)

    return results


def preprocess(img, net_w, net_h):
    # img bgr2rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize img
    img_resize = cv2.resize(img_rgb, (net_w, net_h))

    # norm img
    img_resize = torch.from_numpy(img_resize.transpose((2, 0, 1)))
    img_norm = img_resize.float().div(255).unsqueeze(0)
    return img_norm


def model_init(model_path, S=7, B=2, C=20):
    # load moel
    model = YOLOV1(S=S, B=B, C=C)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


if __name__ == '__main__':
    # load moel
    checkpoint_path = 'weights/yolov1_final.pth'
    S, B, C = 7, 2, 20
    model = model_init(checkpoint_path, S, B, C)

    # params init
    net_w, net_h = 448, 448
    thresh = 0.2
    iou_thresh = 0.4
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    # load img
    img = cv2.imread('demo/person.jpg')
    img_h, img_w, _ = img.shape

    # preprocess
    img_norm = preprocess(img, net_w, net_h)

    # forward
    output = model(img_norm)

    # postprocess
    results = postprocess(output, thresh, S, B, C, img_w, img_h)

    # nms
    results = nms(results, iou_thresh)

    # show
    for i in range(results.shape[0]):
        if results[i][4] > thresh:
            cv2.rectangle(img, (int(results[i][0]), int(results[i][1])), (int(results[i][2]), int(results[i][3])), (0,255,0), 2)
            cv2.putText(img, classes[int(results[i][5])] + '-' + str(round(results[i][4], 4)), (int(results[i][0]), int(results[i][1])), 0, 0.6, (0,255,255), 2)

    # cv2.imwrite('demo.jpg', img)
    cv2.imshow('demo', img)
    cv2.waitKey(0)
