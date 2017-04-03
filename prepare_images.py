import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided

import nn
from settings import COVER_PERCENT

IMG_WIDTH = 1025
IMG_HEIGHT = 523
IMG_LAYERS = 3

SUB_IMG_WIDTH = 48
SUB_IMG_HEIGHT = 48
SUB_IMG_LAYERS = 3
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


def split_into_subimgs(img, sub_img_shape, debug, step=1):
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
    return result_array


def get_labels(labels, result_array_shape, step, sub_img_shape):
    lbl_array = np.zeros(shape=(result_array_shape[0], result_array_shape[1]))
    index = 0
    for i in range(lbl_array.shape[0]):
        for ii in range(lbl_array.shape[1]):
            # Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
            window = nn.Rectangle(ii * step, i * step, ii * step + sub_img_shape[HEIGHT],
                                  i * step + sub_img_shape[WIDTH])
            cover = np.array([compute_covering(window=window,
                                               label=nn.Rectangle(lbl[0], lbl[1], lbl[2], lbl[3])) for lbl in labels])
            is_cover = int(np.any(cover > COVER_PERCENT))

            lbl_array[i, ii] = is_cover
            index += 1
    return lbl_array


def prepare(img_path, labels, debug=False):
    step = 2
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if debug:
        print("Prepare image " + img_path)
        print(img.shape)
        print(labels)
    res_img = img / 255
    res_img = np.array([res_img[:, :, 0], res_img[:, :, 1], res_img[:, :, 2]])

    res = split_into_subimgs(res_img, sub_img_shape=(SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH),
                             step=step, debug=debug)
    lbl_res = get_labels(labels=labels, result_array_shape=res.shape,
                         step=step, sub_img_shape=(SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH))

    return res, lbl_res


def prepare_calibration(img_path, labels, debug=False):
    # Возвращает метки в виде (yn, xn, wn, hn), для калибровки рамки изображения
    # если (x, y) координаты верхенго левого угла и (w, h) соответственно ширина и высота,
    # то новая рамка будет (x - xn * w / wn, y - yn * h / hn), (w / wn, h / hn)
    # TODO: проанализировать данные и подобрать оптимальные yn, xn, wn, hn
    step = 2
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if debug:
        print("Prepare image " + img_path)
        print(img.shape)
        print(labels)
    res_img = img / 255
    res_img = np.array([res_img[:, :, 0], res_img[:, :, 1], res_img[:, :, 2]])

    res = split_into_subimgs(res_img, sub_img_shape=(SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH),
                             step=step, debug=debug)

    lbl_res = []

    return res, lbl_res


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
