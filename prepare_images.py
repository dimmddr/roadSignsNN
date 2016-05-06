import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.lib.stride_tricks import as_strided

from nn import Rectangle

# Max possible size of image
IMG_WIDTH = 1025
IMG_HEIGHT = 523
IMG_LAYERS = 3

SUB_IMG_WIDTH = 48
SUB_IMG_HEIGHT = 48
SUB_IMG_LAYERS = 3
COVER_PERCENT = 0.6
WIDTH = 2
HEIGHT = 1
LAYERS = 0

XMIN = 0
YMIN = 1
XMAX = 2
YMAX = 3


# TODO: переписать либо все с использованием Rectangle namedtuple, либо через numpy. Например с помощью recarray

def compute_covering(window, label):
    dx = min(window.xmax, label.xmax) - max(window.xmin, label.xmin)
    dy = min(window.ymax, label.ymax) - max(window.ymin, label.ymin)
    if (dx >= 0) and (dy >= 0):
        label_cover = dx * dy / ((label.xmax - label.xmin) * (label.ymax - label.ymin))
        window_cover = dx * dy / ((window.xmax - window.xmin) * (window.ymax - window.ymin))
        return max(label_cover, window_cover)
    else:
        return 0


def split_into_subimgs(img, labels, sub_img_shape, debug, step=1):
    shape = (int(np.floor((img.shape[HEIGHT] - sub_img_shape[HEIGHT]) / step)),
             int(np.floor((img.shape[WIDTH] - sub_img_shape[WIDTH]) / step)),
             SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH)
    # shape = (lbl_array.shape[0], SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH)
    result_array = as_strided(img, shape=shape,
                              strides=(
                                  img.strides[1] * step + (img.shape[WIDTH] - sub_img_shape[WIDTH]) % step *
                                  img.strides[2],
                                  img.strides[2] * step,
                                  img.strides[0], img.strides[1], img.strides[2]))

    lbl_array = np.zeros(shape=(result_array.shape[0], result_array.shape[1]))
    index = 0

    coords = dict()
    for i in range(lbl_array.shape[0]):
        for ii in range(lbl_array.shape[1]):
            # Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
            window = Rectangle(ii * step, i * step, ii * step + sub_img_shape[HEIGHT], i * step + sub_img_shape[WIDTH])
            cover = np.array([compute_covering(window=window,
                                               label=Rectangle(lbl[0], lbl[1], lbl[2], lbl[3])) for lbl in labels])
            is_cover = int(np.any(cover > COVER_PERCENT))

            lbl_array[i, ii] = is_cover
            coords[index] = window
            index += 1
    return result_array, lbl_array, coords


def prepare(img_path, labels, debug=False):
    step = 2
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if debug:
        print("Prepare image " + img_path)
        print(img.shape)
        print(labels)
    res_img = img / 255
    res_img = np.array([res_img[:, :, 0], res_img[:, :, 1], res_img[:, :, 2]])

    res, lbl_res, coords = split_into_subimgs(res_img, sub_img_shape=(SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH),
                                              labels=labels, step=step, debug=debug)

    return res, lbl_res, coords


def show_sign(img_path, lbl):
    print(img_path)
    print(lbl)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    cv2.imshow("img", img[lbl[1]:lbl[3], lbl[0]:lbl[2], :])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.rectangle(img, (lbl[0], lbl[1]), (lbl[2], lbl[3]), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_roi(roi_list):
    for roi in roi_list:
        (r, g, b) = (roi[0], roi[1], roi[2])
        roi = cv2.merge((r, g, b))
        cv2.imshow("img", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_rectangles(filename, rectangles_list, show_type='matplotlib'):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    for rect in rectangles_list:
        if rect is not None:
            cv2.rectangle(img, (rect[XMIN], rect[YMIN]), (rect[XMAX], rect[YMAX]), (0, 255, 0), 1)
    if show_type == 'matplotlib':
        (b, g, r) = cv2.split(img)
        img = cv2.merge((r, g, b))
        plt.imshow(img)
        plt.show()
    else:
        cv2.imshow(filename, img)
        cv2.waitKey()


# TODO добавить схранение в отдельный каталог
def save_img_with_rectangles(dataset_path, filename, rectangles_list):
    img = cv2.imread(dataset_path + filename, cv2.IMREAD_UNCHANGED)
    for rect in rectangles_list:
        if rect is not None:
            cv2.rectangle(img, (rect[XMIN], rect[YMIN]), (rect[XMAX], rect[YMAX]), (0, 255, 0), 1)
    cv2.imwrite(dataset_path + "results/" + filename + "_with_rects.jpg", img)


# Probably temp function before I fix localization
def get_roi_from_images(images, img_path):
    res_roi = []
    res_label = []
    label_dict = dict()
    for image in images:
        img = cv2.imread(img_path + image.filename.decode('utf8'), cv2.IMREAD_UNCHANGED)
        for sign in image.signs:
            if sign.label not in label_dict:
                label_dict[sign.label] = len(label_dict)
            (x1, y1, x2, y2) = sign.coord
            roi = img[y1:y2, x1:x2, :]
            res_roi.append(np.array([roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]]))
            res_label.append(label_dict[sign.label])
    return res_roi, res_label, label_dict


def create_synthetic_data(imgs):
    # Create array of size mods [1, 4], step = 0.5
    sizes = np.arange(start=1, stop=4.5, step=0.5)
    total = imgs.shape[0] * sizes.shape[0] * 2  # *2
    res = []
    return imgs


def nms(boxes, overlap_threshold):
    if 0 == len(boxes):
        return []

    boxes = np.array(boxes)

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (xmax - xmin + 1) * (ymax - ymin + 1)
    idxs = np.argsort(ymax)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(xmin[i], xmin[idxs[:last]])
        yy1 = np.maximum(ymin[i], ymin[idxs[:last]])
        xx2 = np.minimum(xmax[i], xmax[idxs[:last]])
        yy2 = np.minimum(ymax[i], ymax[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick], pick
