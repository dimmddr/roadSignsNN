import data_utils
import test_nn
import settings


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

    test_nn.test_neural_net(
        indexes=[5, 15],
        batch_sizes=(5, 15),
        filters=[[5, 15], [25, 30]],
        sizes=((settings.sub_image['layers'], settings.sub_image['height'] // 4, settings.sub_image['width'] // 4),
               (settings.sub_image['layers'], settings.sub_image['height'] // 2, settings.sub_image['width'] // 2),
               (settings.sub_image['layers'], settings.sub_image['height'], settings.sub_image['width']))
    )


if __name__ == '__main__':
    test_all()
