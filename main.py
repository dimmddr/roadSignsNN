import data_utils
from test_nn import annotation_learning_path


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
    data_analysis()


if __name__ == '__main__':
    test_all()
