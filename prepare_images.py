import cv2
import numpy as np
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


def compute_covering(window, label):
    dx = min(window.xmax, label.xmax) - max(window.xmin, label.xmin)
    dy = min(window.ymax, label.ymax) - max(window.ymin, label.ymin)
    if (dx >= 0) and (dy >= 0):
        label_cover = dx * dy / ((label.xmax - label.xmin) * (label.ymax - label.ymin))
        window_cover = dx * dy / ((window.xmax - window.xmin) * (window.ymax - window.ymin))
        return max(label_cover, window_cover)
    else:
        return 0


def split_into_subimgs(img, labels, sub_img_shape, lbl_array, debug, step=1):
    shape = (lbl_array.shape[0], SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH)
    result_array = as_strided(img, shape=shape,
                              strides=(img.strides[2] * step, img.strides[0], img.strides[1], img.strides[2]))
    index = 0

    coords = dict()
    for i in range(0, img.shape[HEIGHT] - sub_img_shape[HEIGHT], step):
        for ii in range(0, img.shape[WIDTH] - sub_img_shape[WIDTH], step):
            # Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
            window = Rectangle(i, ii, i + sub_img_shape[HEIGHT], ii + sub_img_shape[WIDTH])
            cover = np.array([compute_covering(window=window,
                                               label=Rectangle(lbl[1], lbl[0], lbl[3], lbl[2])) for lbl in labels])
            is_cover = int(np.any(cover > COVER_PERCENT))
            if debug and (1 == is_cover):
                roi = img[:, i: i + sub_img_shape[HEIGHT], ii: ii + sub_img_shape[WIDTH]]
                (r, g, b) = (roi[0], roi[1], roi[2])
                roi = cv2.merge((r, g, b))
                cv2.imshow("from labels", roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                roi = result_array[index]
                (r, g, b) = (roi[0], roi[1], roi[2])
                roi = cv2.merge((r, g, b))
                cv2.imshow("from strides", roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            lbl_array[index] = is_cover
            coords[index] = window
            index += 1
        index += int(sub_img_shape[WIDTH] / step)
    return result_array, coords


def prepare(img_path, labels, debug=False):
    step = 2
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if debug:
        print("Prepare image " + img_path)
        print(img.shape)
        print(labels)
    res_img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    res_img = np.array([res_img[:, :, 0], res_img[:, :, 1], res_img[:, :, 2]])

    lbl_res = np.zeros(shape=int(
        (res_img.shape[WIDTH] - SUB_IMG_WIDTH + 1) / step * (res_img.shape[HEIGHT] - SUB_IMG_HEIGHT + 1) / step + (
            SUB_IMG_WIDTH / step * res_img.shape[HEIGHT] / step)))
    res, coords = split_into_subimgs(res_img, sub_img_shape=(SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH),
                                     labels=labels,
                                     lbl_array=lbl_res, step=step, debug=debug)

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


def show_rectangles(filename, rectangles_list):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    for rect in rectangles_list:
        if rect is not None:
            cv2.rectangle(img, (rect.xmin, rect.ymin), (rect.xmax, rect.ymax), (0, 255, 0), 1)
    cv2.imshow(filename, img)
    cv2.waitKey()
