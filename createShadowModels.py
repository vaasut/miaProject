import csv
import os
import random
import shutil

import numpy as np
import tensorflow as tf
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

from EvaluateTensorClassifier import loadModel
from ridgeClassifier import saveModel
from tensorClassifier import buildModel


def createShadowModelTrainingSets(n,folderName):
    for i in range(n):
        newDirectory = "shadowModelSets/model" + str(i) + "/train"
        keywords = ["art_nouveau", "ukiyo_e"]
        for label in range(len(keywords)):
            os.makedirs(newDirectory + "/" + keywords[label])
            imageDirectory = folderName + "/" + keywords[label]
            files = os.listdir(imageDirectory)
            random.shuffle(files)
            files = files[0:25]
            for file in files:
                shutil.copy(imageDirectory + "/" + file,newDirectory + "/" + keywords[label] + '/' + file)
            # print(files)

def createShadowModelTestingSets(n,folderName):
    for i in range(n):
        newDirectory = "shadowModelSets/model" + str(i) + "/test"
        keywords = ["art_nouveau", "ukiyo_e"]
        for label in range(len(keywords)):
            os.makedirs(newDirectory + "/" + keywords[label])
            imageDirectory = folderName + "/" + keywords[label]
            files = os.listdir(imageDirectory)
            random.shuffle(files)
            files = files[0:25]
            for file in files:
                shutil.copy(imageDirectory + "/" + file,newDirectory + "/" + keywords[label] + '/' + file)
            # print(files)

folderName = "artbench-10-imagefolder-split-two/test_two"
# createShadowModelTrainingSets(200,folderName)
# createShadowModelTestingSets(200,folderName)
def buildShadowModels(n):
    for i in range(n):
        print("Building Model: " + str(i))
        buildModel("shadowModelSets/model" + str(i) + "/train","shadowModels/model" + str(i))

# buildShadowModels(200)

def createShadowModelVectors(n):
    classLabels = ["art_nouveau", "ukiyo_e"]
    labels = []
    vectors = []
    for i in range(n):
        print(i)
        for j in range(2):
            labels.append(j)
            if j == 0:
                folder = "test"
            else:
                folder = "train"
            vector = []
            for artClass in classLabels:
                model = loadModel('shadowModels/model' + str(i))
                directory = "shadowModelSets/model" + str(i) + "/" + folder + "/" + artClass
                for filename in os.listdir(directory):
                    try:
                      file_url = directory + "/" + filename
                      img = tf.keras.utils.load_img(file_url, target_size=(64, 64))
                      img_array = tf.keras.utils.img_to_array(img)
                      img_array = tf.expand_dims(img_array, 0) # Create a batch
                      predictions = model.predict(img_array)
                      score = np.array(tf.nn.softmax(predictions[0]))
                      vector.append(score[0])
                    except:
                        continue
            vectors.append(vector)
    return labels, vectors

def writeVectorsToFile(vectors,filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(vectors)

def readVectorsFromFile(filename):
    vectors = []
    with open(filename) as f:
        reader = csv.reader(f)
        num = 0
        for row in reader:
            vector = []
            for i in row:
                vector.append(float(i))
            if len(vector) == 50:
                vectors.append(vector)
            else:
                print(num)
            num += 1
    return vectors

def readLabelsFromFile(filename):
    labels = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
          for i in range(len(row)):
              if i != 85:
                labels.append(int(i)%2)
    return labels


def getTestImages(folderName,n):
    """ Function returns an array of n random image names from the given folder."""
    testImages = []
    files = os.listdir(folderName)
    random.shuffle(files)
    testImages = files[0:n]
    return [folderName + "/" + i for i in testImages]

def attackTargetModel(imageSet):
    vector = []
    model = loadModel('saved_model/myTensorFlowClassifierModelMiniTwoNew')
    for file_url in imageSet:
        try:
            img = tf.keras.utils.load_img(file_url, target_size=(64, 64))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch
            predictions = model.predict(img_array)
            score = np.array(tf.nn.softmax(predictions[0]))
            vector.append(score[0])
        except:
            continue
    return vector

# labels,vectors = createShadowModelVectors(200)
# writeVectorsToFile(vectors,"vectorFile.csv")
# writeVectorsToFile(labels,"labelFile.csv")
vectors = np.matrix(readVectorsFromFile("vectorFile.csv"))
labels = readLabelsFromFile("labelFile.csv")
print(vectors)
classifierModel = RidgeClassifier(n_neighbors=10)
classifierModel.fit(vectors,labels)
testImages = getTestImages("specificArtists/in-training/okumura",50)
vector = attackTargetModel(testImages)
# print(vector)
prediction = classifierModel.predict_proba([vector])
print(prediction)
# saveModel(classifierModel,"myAttackModel.joblib")