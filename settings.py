# Negative to positive ratio
NEGATIVE_MULTIPLIER = 2

# Images count for neural nets training
FIRST_IND = 75
SECOND_IND = 250

# Batch sizes for neural nets
FIRST_BATCH_SIZE = 50
SECOND_BATCH_SIZE = 30
THIRD_BATCH_SIZE = 30
# Learning ratio
ALFA = 0.01
# Filter counts
FIRST_FILTER_NUMBERS = 100
SECOND_FILTER_NUMBERS = 200
THIRD_FILTER_NUMBERS = 200
# Filter sizes
FIRST_FILTER_SIZE = (5, 5)
SECOND_FILTER_SIZE = (7, 7)
THIRD_FILTER_SIZE = (7, 7)

# Size of image window in picture
sub_image = {
    'width': 48,
    'height': 48,
    'layers': 3
}

NET_12, NET_24, NET_48, NET_12_CALIBRATION, NET_24_CALIBRATION = list(range(5))

DATASET_PATH = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/"
ANNOTATION_LEARNING_PATH = DATASET_PATH + 'learningAnnotations.csv'
ANNOTATION_TEST_PATH = DATASET_PATH + 'testAnnotations.csv'
# dataset_path = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/"
# annotation_path = dataset_path + 'allAnnotations.csv'
# dataset_path = "c:/_Hive/_diploma/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/vid0/frameAnnotations-vid_cmp2.avi_annotations/"
# annotation_path = dataset_path + 'frameAnnotations.csv'
