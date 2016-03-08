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


def compute_covering(window, label):
    dx = min(window.xmax, label.xmax) - max(window.xmin, label.xmin)
    dy = min(window.ymax, label.ymax) - max(window.ymin, label.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy / ((label.xmax - label.xmin) * (label.ymax - label.ymin))
    else:
        return 0


def split_into_subimgs(img, lbl, sub_img_shape, result_array, lbl_array, step=1):
    index = 0
    for i in range(0, img.shape[0] - sub_img_shape[0], step):
        for ii in range(0, img.shape[1] - sub_img_shape[1], step):
            # if img.shape[0] - i >= sub_img_shape[0] or img.shape[1] - ii >= sub_img_shape[1]:
            result_array[index] = img[i:i + sub_img_shape[0], ii:ii + sub_img_shape[1], :]
            lbl_array[index] = int(compute_covering(Rectangle(i, ii, i + sub_img_shape[0], ii + sub_img_shape[1]),
                                                    Rectangle(lbl[0], lbl[1], lbl[2], lbl[3])) > COVER_PERCENT)
            index += 1


def prepare(img_path, lbl):
    step = 2
    lbl_res = np.zeros(shape=((IMG_WIDTH - SUB_IMG_WIDTH + 1) / step * (IMG_HEIGHT - SUB_IMG_HEIGHT + 1) / step,))
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    res_img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    res = np.zeros(
        shape=(
            (res_img.shape[0] - SUB_IMG_WIDTH + 1) / step * (res_img.shape[1] - SUB_IMG_HEIGHT + 1) / step,
            SUB_IMG_HEIGHT, SUB_IMG_WIDTH, SUB_IMG_LAYERS),
        dtype=np.dtype(float))
    split_into_subimgs(res_img, sub_img_shape=(SUB_IMG_HEIGHT, SUB_IMG_WIDTH, SUB_IMG_LAYERS),
                       result_array=res, lbl=lbl, lbl_array=lbl_res, step=step)

    return res, lbl_res

# x1 = image_data['Upper_left_corner_X'][0]
# x2 = image_data['Lower_right_corner_X'][0]
# y1 = image_data['Upper_left_corner_Y'][0]
# y2 = image_data['Lower_right_corner_Y'][0]
# roi = img[y1:y2, x1:x2]
# fname = dataset_path + "signs/qwer.png"
# print(fname)
# cv2.imwrite(fname, roi)
# cv2.imshow('image', roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# lengths = train['Lower_right_corner_X'] - train['Upper_left_corner_X']
# heights = train['Lower_right_corner_Y'] - train['Upper_left_corner_Y']
# for img_rec in train:
#     img = cv2.imread(dataset_path + img_rec['Filename'].decode('utf8'), cv2.IMREAD_UNCHANGED)
#     x1 = train['Upper_left_corner_X'][0]
#     x2 = train['Lower_right_corner_X'][0]
#     y1 = train['Upper_left_corner_Y'][0]
#     y2 = train['Lower_right_corner_Y'][0]
#     roi = img[y1:y2, x1:x2]
#     fname = dataset_path + "signs/" + img_rec['Filename'].decode('utf8')
#     print(fname)
#     cv2.imwrite(fname, roi)
