import cv2
import numpy as np

from nn import Rectangle

# Max possible size of image
IMG_WIDTH = 1025
IMG_HEIGHT = 523
IMG_LAYERS = 3

SUB_IMG_WIDTH = 48
SUB_IMG_HEIGHT = 48
SUB_IMG_LAYERS = 3
COVER_PERCENT = 0.6
WIDTH = 1
HEIGHT = 2
LAYERS = 0


def compute_covering(window, label):
    dx = min(window.xmax, label.xmax) - max(window.xmin, label.xmin)
    dy = min(window.ymax, label.ymax) - max(window.ymin, label.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy / ((label.xmax - label.xmin) * (label.ymax - label.ymin))
    else:
        return 0


def split_into_subimgs(img, lbl, sub_img_shape, result_array, lbl_array, step=1):
    index = 0
    for i in range(0, img.shape[WIDTH] - sub_img_shape[WIDTH], step):
        for ii in range(0, img.shape[HEIGHT] - sub_img_shape[HEIGHT], step):
            result_array[index] = img[:, i:i + sub_img_shape[WIDTH], ii:ii + sub_img_shape[HEIGHT]]
            lbl_array[index] = int(compute_covering(Rectangle(i, ii, i + sub_img_shape[0], ii + sub_img_shape[1]),
                                                    Rectangle(lbl[0], lbl[1], lbl[2], lbl[3])) > COVER_PERCENT)
            index += 1


def prepare(img_path, lbl):
    print("Prepare image " + img_path)
    step = 2
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    res_img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    res_img = np.array([res_img[:, :, 0], res_img[:, :, 1], res_img[:, :, 2]])
    res = []
    lbl_res = np.zeros(shape=int(
        (res_img.shape[WIDTH] - SUB_IMG_WIDTH + 1) / step * (res_img.shape[HEIGHT] - SUB_IMG_HEIGHT + 1) / step))
    split_into_subimgs(res_img, sub_img_shape=(SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH),
                       result_array=res, lbl=lbl, lbl_array=lbl_res, step=step)

    return res, lbl_res
