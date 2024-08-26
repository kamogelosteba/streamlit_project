import os
import platform
import shutil
import time
from pathlib import Path
from collections import deque
import numpy as np

import cv2
import torch
from numpy import random

from norfair.drawing.draw_boxes import draw_boxes
from retinaface import RetinaFace

from norfair.drawing import Paths
from norfair import Detection, Tracker, draw_tracked_boxes, Paths, draw_boxes

font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
fontScale = 1
fontColor = (0,255,0)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)

    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        cv2.rectangle(img, c1, c2, color, tl, cv2.LINE_AA)  # not filled

        ##text Class
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


        # Distance function
def centroid_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def get_norfair_detections(detections):
    face_norfair_detections = []

    for bbox in detections:
        bbox = np.array(bbox).reshape(2, 2)
        face_norfair_detections.append(Detection(points=bbox))

    return face_norfair_detections


def convert_bboxes_to_absolute(obj):
  bbox = obj.estimate
  x1, y1 = np.absolute(bbox[0]).tolist()
  x1, y1 = int(x1), int(y1)
  x2, y2 = np.absolute(bbox[1]).tolist()
  x2, y2 = int(x2), int(y2)

  return (x1, y1, x2, y2)

