import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import tensorflow as tf
import re

# from google.colab.patches import cv2_imshow

def test():
  return "hello from lpr"

###-------------------------------------------------LPR-----------------------------------------------
class LPR:
  def __init__(self,modelConfiguration = "darknet-yolov3.cfg",modelWeights = "lapi.weights"):
   
    self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    self.confThreshold = 0.5  
    self.nmsThreshold = 0.4
    self.inpWidth = 416
    self.inpHeight = 416

  def read_img(self,path):
    frame = cv.imread(path)
    # cv.imshow("frame",frame)
    return frame

  
  def getOutputsNames(self):
    layersNames = self.net.getLayerNames()
    # Get the names of the output layers and unconnected layer outputs
    return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]  

  def drawPred(self,frame,conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

  def detect_plate(self,frame):
    #inp : frame (img object)
    blob = cv.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
    self.net.setInput(blob)
    
    #run img through network forward pass
    outputs = self.net.forward(self.getOutputsNames())

    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    for output in outputs:
      for detection in output:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        #get max confidence

        if confidence > self.confThreshold:
          center_x = int(detection[0] * frameWidth)
          center_y = int(detection[1] * frameHeight)
          width = int(detection[2] * frameWidth)
          height = int(detection[3] * frameHeight)
          left = int(center_x - width / 2)
          top = int(center_y - height / 2)
          classIds.append(classId)
          confidences.append(float(confidence))
          boxes.append([left, top, width, height])

    #non maximum suppression to eliminate overlapping boxes
    
    indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
    plate_co_ordinates = []
    for index in indices:
        i = index[0]
        #get coordinates of the box after NMS
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        plate_co_ordinates.append([top, top + height, left,left + width])
    return plate_co_ordinates

###-------------------------------------------------LPR-----------------------------------------------




###-------------------------------------------------CR------------------------------------------------
class CR:
  def __init__(self,modelFile = 'character_recognition.h5'):
    self.charRecogModel = tf.keras.models.load_model(modelFile)
    self.loaded = tf.saved_model.load('./WEIGHTS/CNN/')
    self.infer = self.loaded.signatures["serving_default"]


  def predict_tess(image):
    text = pytesseract.image_to_string(image)
    text = re.sub(r'[^A-Z0-9a-z]','',text)
    text = re.sub(r'[oO]','0',text)
    return text

  def predict_char_saved(self,img):
    # loaded = tf.saved_model.load('./')
    # infer = loaded.signatures["serving_default"]

    map = { 0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
    11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
    21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
    30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z' }
    blackAndWhiteChar=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blackAndWhiteChar = cv.resize(blackAndWhiteChar,(75,100))
    image = blackAndWhiteChar.reshape((1, 100,75, 1))
    image = image / 255.0

    img_inp = tf.constant(image,dtype=np.float32)
    predictions = np.array(self.infer(img_inp)['dense_1'])

    char = np.argmax(predictions)
    return map[char]

  # def predict_char(self,img):
  #   map = { 0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
  #   11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
  #   21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
  #   30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z' }
  #   blackAndWhiteChar=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  #   blackAndWhiteChar = cv.resize(blackAndWhiteChar,(75,100))
  #   image = blackAndWhiteChar.reshape((1, 100,75, 1))
  #   image = image / 255.0
  #   predictions = self.charRecogModel.predict(image)
  #   char = np.argmax(predictions)
  #   return map[char]

  def auto_canny(self,image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    # return the image with edges
    return edged


  def opencvReadPlate(self,img):
    charList=[]
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    thresh_inv = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,39,1)
    edges = self.auto_canny(thresh_inv)
    ctrs, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv.boundingRect(ctr)[0])

    img_area = img.shape[0]*img.shape[1]

    for i, ctr in enumerate(sorted_ctrs):
      x, y, w, h = cv.boundingRect(ctr)
      #add padding

      x1 = int(x - 3)
      y1 = int(y - 3)
      x2 = int((x+w) + 3)
      y2 = int((y+h) + 3)

      roi_area = w*h
      non_max_sup = roi_area/img_area

      if((non_max_sup >= 0.015) and (non_max_sup < 0.09)):
        if ((h>1*w) and (4*w>=h)):
            # char = img[y:y+h,x:x+w]
          char = img[y1:y2,x1:x2]
          charList.append(self.predict_char_saved(char))
    return charList

  def opencvReadPlate2(self,img):
    return self.predict_tess(img)


###-------------------------------------------------CR------------------------------------------------

###-----------------------------------------execution code--------------------------------------------

# platedetector = LPR(modelConfiguration="./WEIGHTS/darknet-yolov3.cfg",modelWeights = "./WEIGHTS/lapi.weights")
# charRecognizer = CR(modelFile='./WEIGHTS/character_recognition.h5')

# img = platedetector.read_img('./images/car3.jpg')
# plate_coor = platedetector.detect_plate(img)
# if(len(plate_coor)):
# # cv.imshow("chars",img[plate_coor[0][0]:plate_coor[0][1],plate_coor[0][2]:plate_coor[0][3]])
#   print(charRecognizer.opencvReadPlate(img[plate_coor[0][0]:plate_coor[0][1],plate_coor[0][2]:plate_coor[0][3]]))
# print("done")

###-----------------------------------------save model code--------------------------------------------

# charRecogModel = tf.keras.models.load_model('./WEIGHTS/character_recognition.h5')
# tf.saved_model.save(charRecogModel,'./WEIGHTS/CNN/')  ### to save model
# print("SAVED")