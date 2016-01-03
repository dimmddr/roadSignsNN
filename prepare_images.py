import cv2
import numpy as np
import numpy.lib.recfunctions

dataset_path = "h:/_diplomaData/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/"
annotation_path = dataset_path + 'allAnnotations.csv'
negatives_path = dataset_path + "negatives.dat"
rnd = np.random.RandomState(16)

# Read data from files
image_data = np.genfromtxt(annotation_path, delimiter=';', names=True, dtype=None)
negatives = np.genfromtxt(negatives_path, dtype=None)

# Set size variables. I set amount of negatives about 1/3 of other foto
train_set_size = np.floor(len(image_data) * 0.75)
test_set_size = len(image_data) - train_set_size
neg_train_size = np.floor(train_set_size / 3)
neg_test_size = np.floor(test_set_size / 3)

# Get random sets and concat it with random negatives
np.random.shuffle(image_data)
train_set, test_set = image_data[:train_set_size], image_data[train_set_size:]
neg_train = np.random.choice(negatives, size=neg_train_size, replace=False).view(dtype=[('Filename', '|S28')])
neg_test = np.random.choice(negatives, size=neg_test_size, replace=False).view(dtype=[('Filename', '|S28')])
train_set = numpy.lib.recfunctions.stack_arrays((train_set, neg_train), autoconvert=True, usemask=False)
test_set = numpy.lib.recfunctions.stack_arrays((test_set, neg_test), autoconvert=True, usemask=False)
h, s, v = cv2.split(hsv_img)
x1 = image_data['Upper_left_corner_X'][0]
x2 = image_data['Lower_right_corner_X'][0]
y1 = image_data['Upper_left_corner_Y'][0]
y2 = image_data['Lower_right_corner_Y'][0]
roi = img[y1:y2, x1:x2]
fname = dataset_path + "signs/qwer.png"
print(fname)
cv2.imwrite(fname, roi)
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
