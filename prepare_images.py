import cv2
import numpy as np

IMG_WIDTH = 1025
IMG_HEIGHT = 523
IMG_LAYERS = 3

SUB_IMG_WIDTH = 48
SUB_IMG_HEIGHT = 48
SUB_IMG_LAYERS = 3


def split_into_subimgs(img, sub_img_shape, result_array, step=1):
    index = 0
    for i in range(0, img.shape[0] - sub_img_shape[0], step):
        for ii in range(0, img.shape[1] - sub_img_shape[1], step):
            result_array[index] = img[i:sub_img_shape[0], ii:sub_img_shape[1], :]
            index += 1


def prepare(img_path):
    step = 2
    res = np.zeros(
        shape=(
            (IMG_WIDTH - SUB_IMG_WIDTH + 1) / step * (IMG_HEIGHT - SUB_IMG_HEIGHT + 1) / step,
            SUB_IMG_HEIGHT, SUB_IMG_WIDTH, SUB_IMG_LAYERS),
        dtype=np.dtype(float))
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    res_img = cv2.copyMakeBorder(src=img,
                                 top=0, left=0, bottom=IMG_HEIGHT - img.shape[0], right=IMG_WIDTH - img.shape[1],
                                 borderType=cv2.BORDER_CONSTANT, value=0)
    res_img /= 256
    split_into_subimgs(res_img, sub_img_shape=(SUB_IMG_HEIGHT, SUB_IMG_WIDTH, SUB_IMG_LAYERS),
                       result_array=res, step=step)

    return res

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
