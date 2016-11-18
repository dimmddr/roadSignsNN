import data_utils
import test_nn
import settings


def data_analysis():
    data_utils.analyse_sign_frame_size_fluctuations(settings.ANNOTATION_LEARNING_PATH, "sign_frame_size_fluctuations")


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
    # TODO: Make params like filters sizes and batch sizes into dict for every network

    test_nn.test_neural_net(
        indexes=[1, 2, 3],
        batch_sizes=(5, 15, 15),
        filters=[[15, (3, 3)], [24, (6, 6)], [48, (12, 12)]],
        sizes=((settings.sub_image['layers'], settings.sub_image['height'] // 4, settings.sub_image['width'] // 4),
               (settings.sub_image['layers'], settings.sub_image['height'] // 2, settings.sub_image['width'] // 2),
               (settings.sub_image['layers'], settings.sub_image['height'], settings.sub_image['width']))
    )


if __name__ == '__main__':
    test_all()
