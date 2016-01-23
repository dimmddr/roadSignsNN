import cv2

IMG_WIDTH = 1025
IMG_HEIGHT = 523


def prepare(path_to_img):
    img = cv2.imread(path_to_img, cv2.IMREAD_UNCHANGED)
    res = cv2.copyMakeBorder(src=img,
                             top=0, left=0, bottom=IMG_HEIGHT - img.shape[0], right=IMG_WIDTH - img.shape[1],
                             borderType=cv2.BORDER_CONSTANT, value=0)
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
