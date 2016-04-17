from unittest import TestCase

import cv2
import numpy as np

import prepare_images
from image import Image


class TestPrepareImages(TestCase):
    def test_prepare(self):
        dataset_path = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/vid0/frameAnnotations-vid_cmp2.avi_annotations/"
        annotation_path = dataset_path + 'frameAnnotations.csv'
        image_data = np.genfromtxt(annotation_path, delimiter=';', names=True, dtype=None)
        files = dict()
        for image in image_data:
            filepath = image['Filename']
            if filepath not in files:
                img = Image(filepath)
                img.add_sign(label=image['Annotation_tag'],
                             coordinates=(image['Upper_left_corner_X'], image['Upper_left_corner_Y'],
                                          image['Lower_right_corner_X'], image['Lower_right_corner_Y']))
                files[filepath] = img
            else:
                files[filepath].add_sign(label=image['Annotation_tag'],
                                         coordinates=(image['Upper_left_corner_X'], image['Upper_left_corner_Y'],
                                                      image['Lower_right_corner_X'], image['Lower_right_corner_Y']))
        images = np.array(list(files.keys()))
        images.sort()
        lbl = np.array([files.get(key).get_coordinates() for key in images])
        imgs, lbls, coords = prepare_images.prepare(dataset_path + images[0].decode('utf8'), lbl[0])
        test_img = cv2.imread(dataset_path + images[0].decode('utf8'), cv2.IMREAD_UNCHANGED)
        test_img = cv2.normalize(test_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        for j in range(lbls.shape[0]):
            if coords[j] is not None:
                # Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
                (x1, y1, x2, y2) = coords[j]
                test_img_roi = np.array(
                    [test_img[y1:y2, x1:x2, 0], test_img[y1:y2, x1:x2, 1], test_img[y1:y2, x1:x2, 2]])
                prepare_images.show_roi([test_img_roi, imgs[j]])
                self.assertTrue(np.allclose(imgs[j], test_img_roi), msg="In imgs[{}]".format(j))
