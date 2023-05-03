import os
import random

import numpy as np
import tensorflow as tf
from numpy import mean

def loadModel(modelLocation):
  new_model = tf.keras.models.load_model(modelLocation)
  return new_model


def getTestImages(folderName,n):
  """ Function returns an array of n random image names from the given folder."""
  testImages = []
  labels = ["art_nouveau", "ukiyo_e"]
  for label in labels:
    files = os.listdir(folderName + label + "/")
    random.shuffle(files)
    testImages.append(files[0:n])
  return testImages

def getScore(model,directory):
  confusionMatrix = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
  class_names = [""]
  for folder in range(1):
    scores = []
    for filename in os.listdir(directory):
      file_url = directory + class_names[folder] + "/" + filename

      img = tf.keras.utils.load_img(file_url, target_size=(64, 64))
      img_array = tf.keras.utils.img_to_array(img)
      img_array = tf.expand_dims(img_array, 0) # Create a batch

      predictions = model.predict(img_array)
      score = tf.nn.softmax(predictions[0])
      if np.argmax(score) == 0:
        scores.append(100 * np.max(score))
      confusionMatrix[folder][np.argmax(score)] += 1
    print(mean(scores))
  print(confusionMatrix)


# nouveau = ["/in-training/anna", "/in-training/henri", "/in-training/zinaida", "/not-in-training/benois", "/not-in-training/leon",
#  "/not-in-training/virginia"]
# ukiyo = ["/in-training/hishikawa","/in-training/isoda","/in-training/okumura","/not-in-training/hasegawa","/not-in-training/kubo","/not-in-training/yashima"]
# model = loadModel('saved_model/myTensorFlowClassifierModelMiniTwoNewWithout')
# for artist in ukiyo:
#   print(artist)
#   getScore(model,"specificArtists" + artist)
