from lpr import *

platedetector = LPR(modelConfiguration="./WEIGHTS/darknet-yolov3.cfg",modelWeights = "./WEIGHTS/lapi.weights")
charRecognizer = CR(modelFile='./WEIGHTS/character_recognition.h5')

img = platedetector.read_img('./images/car5.jpg')
plate_coor = platedetector.detect_plate(img)
# cv.imshow("chars",img[plate_coor[0][0]:plate_coor[0][1],plate_coor[0][2]:plate_coor[0][3]])
print(charRecognizer.opencvReadPlate(img[plate_coor[0][0]:plate_coor[0][1],plate_coor[0][2]:plate_coor[0][3]]))

print(charRecognizer.opencvReadPlate(img[plate_coor[0][0]:plate_coor[0][1],plate_coor[0][2]:plate_coor[0][3]]))