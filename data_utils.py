import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

        # Annotation tag;Upper left corner X;Upper left corner Y;Lower right corner X;Lower right corner Y;Occluded,On another road
