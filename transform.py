import numpy as np
import cv2


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
    # np.zeros make array of specified shape
    rect = np.zeros((4, 2), 'float32')
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    # np.sum summs the array if axis =1 row-wise else col-wise
    s = np.sum(pts, axis=1)
    # top-left
    rect[0] = pts[np.argmin(s)]
    # bottom-right
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    # top-righ
    rect[1] = pts[np.argmin(diff)]
    # bottom-left
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    # make an rectangle an order pts in [tl, tr, br, bl]
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width as max(w1,w2)
    # w1 = dist b/w tl, tr
    w1 = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2)
    w2 = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2)
    # maxwidth = max w1,w2
    maxW = int(max(w1, w2))

    # compute the height as max(h1,h2)
    # h1 = dist b/w tl, bl
    h1 = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2)
    h2 = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2)
    # maxheight = max h1,h2
    maxH = int(max(h1, h2))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order

    dst = np.array([[0, 0], [maxW-1, 0], [maxW-1, maxH-1],
                    [0, maxH-1]], dtype='float32')

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))

    return warped
