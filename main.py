import test_nn
import settings
import utils


def data_analysis():
    utils.analyse_sign_frame_size_fluctuations(settings.ANNOTATION_LEARNING_PATH, "sign_frame_size_fluctuations")


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
    neural_nets_params = {
        'net_12': {
            'name': 'First location net',
            'indexes': 1,
            'batch_size': 5,
            'filters': [15, (3, 3)],
            'sizes': (settings.sub_image['layers'], settings.sub_image['height'] // 4, settings.sub_image['width'] // 4)
        },
        'net_24': {
            'name': 'Second location net',
            'indexes': 2,
            'batch_size': 15,
            'filters': [24, (6, 6)],
            'sizes': (settings.sub_image['layers'], settings.sub_image['height'] // 2, settings.sub_image['width'] // 2)
        },
        'net_48': {
            'name': 'Third location net',
            'indexes': 3,
            'batch_size': 15,
            'filters': [48, (12, 12)],
            'sizes': (settings.sub_image['layers'], settings.sub_image['height'], settings.sub_image['width'])
        },
        'calibration_net_24': {
            'name': 'Third location net',
            'indexes': 3,
            'batch_size': 15,
            'filters': [48, (12, 12)],
            'sizes': (settings.sub_image['layers'], settings.sub_image['height'], settings.sub_image['width'])
        },
        'calibration_net_48': {
            'name': 'Third location net',
            'indexes': 3,
            'batch_size': 15,
            'filters': [48, (12, 12)],
            'sizes': (settings.sub_image['layers'], settings.sub_image['height'], settings.sub_image['width'])
        }
    }

    test_nn.test_neural_net(neural_nets_params=neural_nets_params)


if __name__ == '__main__':
    test_all()
