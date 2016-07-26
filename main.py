import data_utils
from test_nn import *


def data_analysis():
    data_utils.analyse_sign_frame_size_fluctuations(annotation_learning_path, "sign_frame_size_fluctuations")


def test_all():
    # print("test learning size")
    # test_neural_net_learning_size(start_size=20, end_size=100, step=5, debug=True)
    # print("test batch size")
    # res = test_neural_net(debug=True)
    # print(res)
    # print("Test load parameters")
    # test_load_params()
    # print("Test Classification")
    # test_classification()
    # data_analysis()

    # temp
    SUB_IMG_WIDTH = 48
    SUB_IMG_HEIGHT = 48
    SUB_IMG_LAYERS = 3
    test_neural_net(indexes=[5, 15], batch_sizes=(5, 15), filters=[[5, 15], [25, 30]],
                    sizes=((SUB_IMG_LAYERS, SUB_IMG_HEIGHT // 4, SUB_IMG_WIDTH // 4),
                           (SUB_IMG_LAYERS, SUB_IMG_HEIGHT // 2, SUB_IMG_WIDTH // 2),
                           (SUB_IMG_LAYERS, SUB_IMG_HEIGHT, SUB_IMG_WIDTH)))

if __name__ == '__main__':
    test_all()
