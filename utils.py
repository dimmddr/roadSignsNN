import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import settings


def analyse_sign_frame_size_fluctuations(annotation_path, output_file):
    with open(output_file, 'w') as outp:
        raw_data = pd.read_csv(annotation_path, delimiter=';')
        outp.write("Analyse frame size fluctuations.\n")
        data = pd.DataFrame()
        data['width'] = raw_data['Lower right corner X'] - raw_data['Upper left corner X']
        data['height'] = raw_data['Lower right corner Y'] - raw_data['Upper left corner Y']
        outp.write("Minimum width = {}, minimum height = {}\n".format(data['width'].min(), data['height'].min()))
        outp.write("Maximum width = {}, maximum height = {}\n".format(data['width'].max(), data['height'].max()))

        matplotlib.rcdefaults()
        matplotlib.rcParams['font.family'] = 'fantasy'
        matplotlib.rcParams['font.fantasy'] = 'Times New Roman', 'Ubuntu', 'Arial', 'Tahoma', 'Calibri'
        matplotlib.rcParams.update({'font.size': 18})

        hist, bins = np.histogram(data['width'], bins=range(data['width'].min(), data['width'].max(), 5))
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.title("Ширина дорожных знаков")
        plt.xlabel("Ширина")
        plt.ylabel("Сколько раз встречалась")
        plt.xticks(bins, bins)
        plt.show()

        hist, bins = np.histogram(data['height'], bins=range(data['width'].min(), data['width'].max(), 5))
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.title("Высота дорожных знаков")
        plt.xlabel("Высота")
        plt.ylabel("Сколько раз встречалась")
        plt.xticks(bins, bins)
        plt.show()

        # Annotation tag;
        # Upper left corner X;Upper left corner Y;Lower right corner X;Lower right corner Y;Occluded,On another road


def nms(boxes):
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
                                               np.where(overlap > settings.COVER_PERCENT)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick], pick
