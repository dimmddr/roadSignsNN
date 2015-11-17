import cv2

path_dataset = "h:/_diplomaData/LISA Traffic Sign Dataset/signDatabasePublicFramesOnly/"
path_file = path_dataset + "vid6/frameAnnotations-MVI_0071.MOV_annotations/pedestrian_1323896918.avi_image0.png"

img = cv2.imread(path_file, cv2.IMREAD_UNCHANGED)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
