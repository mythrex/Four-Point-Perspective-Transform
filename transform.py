import numpy as np
import cv2


def order_points(pts):
    # np.zeros make array of specified shape
    rect = np.zeros((4, 2), 'float32')
    # np.sum summs the array if axis =1 row-wise else col-wise
    s = np.sum(pts, axis=1)
    # top-left
    rect[0] = pts[np.argmin(s)]
    # bottom-right
    rect[2] = pts[np.argmax(s)]
    return rect
