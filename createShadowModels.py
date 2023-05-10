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
        newDirectory = "biggerShadowModelSets/model" + str(i) + "/train"
        keywords = ["art_nouveau", "ukiyo_e"]
        for label in range(len(keywords)):
            os.makedirs(newDirectory + "/" + keywords[label])
            imageDirectory = folderName + "/" + keywords[label]
            files = os.listdir(imageDirectory)
            random.shuffle(files)
            files = files[0:500]
            for file in files:
                shutil.copy(imageDirectory + "/" + file,newDirectory + "/" + keywords[label] + '/' + file)
            # print(files)

def createShadowModelTestingSets(n,folderName):
    for i in range(n):
        newDirectory = "biggerShadowModelSets/model" + str(i) + "/test"
        keywords = ["art_nouveau", "ukiyo_e"]
        for label in range(len(keywords)):
            os.makedirs(newDirectory + "/" + keywords[label])
            imageDirectory = folderName + "/" + keywords[label]
            files = os.listdir(imageDirectory)
            random.shuffle(files)
            files = files[0:500]
            for file in files:
                shutil.copy(imageDirectory + "/" + file,newDirectory + "/" + keywords[label] + '/' + file)
            # print(files)

def buildDataSet():
    folderName = "artbench-10-imagefolder-split-two/train_two"
    createShadowModelTrainingSets(10,folderName)
    folderName = "artbench-10-imagefolder-split-two/test_two"
    createShadowModelTestingSets(10,folderName)

def buildShadowModels(n):
    for i in range(n):
        print("Building Model: " + str(i))
        buildModel("biggerShadowModelSets/model" + str(i) + "/train","biggerShadowModels/model" + str(i))

# buildShadowModels(10)

def createShadowModelVectors(n):
    classLabels = ["art_nouveau", "ukiyo_e"]
    labels = []
    vectors = []
    for i in range(n):
        print(i)
        for j in range(2):
            # labels.append(j)
            if j == 0:
                folder = "test"
            else:
                folder = "train"
            for artClass in range(len(classLabels)):
                model = loadModel('biggerShadowModels/model' + str(i))
                directory = "biggerShadowModelSets/model" + str(i) + "/" + folder + "/" + classLabels[artClass]
                for filename in os.listdir(directory):
                    vector = []
                    try:
                      file_url = directory + "/" + filename
                      img = tf.keras.utils.load_img(file_url, target_size=(64, 64))
                      img_array = tf.keras.utils.img_to_array(img)
                      img_array = tf.expand_dims(img_array, 0) # Create a batch
                      predictions = model.predict(img_array)
                      score = np.array(tf.nn.softmax(predictions[0]))
                      vector.append(artClass)
                      vector.append(score[0])
                      vector.append(score[1])
                      # if artClass == "art_nouveau":
                      #   vector.append(score[0])
                      # else:
                      #   vector.append(score[1])
                      vectors.append(vector)
                      labels.append(j)
                    except Exception as e:
                        print(e)
                        continue

    return labels, vectors

def writeVectorsToFile(vectors,filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(vectors)

def readVectorsFromFile(filename, c):
    vectors = []
    labels = []
    with open(filename) as f:
        reader = csv.reader(f)
        num = 0
        for row in reader:
            vector = []
            for i in row:
                vector.append(float(i))
            if len(vector) == 3 and vector[0] == c:
                labels.append(int(num%2000/1000))
                vectors.append(vector)
            num += 1
    return vectors, labels

def getTestImages(folderName,n):
    """ Function returns an array of n random image names from the given folder."""
    testImages = []
    files = os.listdir(folderName)
    random.shuffle(files)
    testImages = files[0:n]
    return [folderName + "/" + i for i in testImages]

def attackTargetModel(imageSet):
    vectors = []
    model = loadModel('saved_model/myTensorFlowClassifierModelDropoutMiniTwoNew')
    for file_url in imageSet:
        try:
            vector = []
            img = tf.keras.utils.load_img(file_url, target_size=(64, 64))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch
            predictions = model.predict(img_array)
            score = np.array(tf.nn.softmax(predictions[0]))
            vector.append(1)
            vector.append(score[0])
            vector.append(score[1])
            vectors.append(vector)
        except:
            continue
    return vectors

def writeAttackDatasetToFile():
    labels,vectors = createShadowModelVectors(10)
    print(vectors)
    writeVectorsToFile(vectors,"vectorFileBiggerModels.csv")
    writeVectorsToFile([labels],"labelFileBiggerModels.csv")

def getAttackModelResults(testImagesDirectory,csplit):
    vectors, labels = readVectorsFromFile("vectorFileFull.csv",csplit)
    print(np.matrix(vectors))
    print(labels)
    classifierModel = KNeighborsClassifier(n_neighbors=3)
    classifierModel.fit(vectors,labels)
    testImages = getTestImages(testImagesDirectory,200)
    vectors = attackTargetModel(testImages)
    prediction = classifierModel.predict(vectors)
    print(prediction)
    print(sum(prediction))
    print(len(prediction))

getAttackModelResults("artbench-10-imagefolder-split-two/train_two/ukiyo_e",1)