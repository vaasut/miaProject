import os
import csv
import random

from PIL import Image
from os import listdir

def invertKey(key):
    for i in range(len(key)):
        key[i] = abs(key[i]-1)
    return key

def convertKeyToString(key):
    result = ""
    for i in key:
        result += str(i)
    return result

def fileKey(folderName,k):
    if (k % 2 == 1):
        k += 1
    temp = int(k/2)
    key = [1] * temp + [0] * temp
    dataset = []
    keywords = ["art_nouveau", "baroque", "expressionism", "impressionism", "post_impressionism", "realism",
                "renaissance", "romanticism", "surrealism", "ukiyo_e"]
    for label in range(len(keywords)):
        print("Starting " + keywords[label])
        files = os.listdir(folderName + "/" + keywords[label])
        random.shuffle(files)
        for i in range(len(files)):
            if (i % 2 == 0):
                random.shuffle(key)
            else:
                key = invertKey(key)
            row = [label, folderName + keywords[label] + "/" + files[i], convertKeyToString(key)]
            dataset.append(row)
    with open("fileModelKey.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataset)

def createDataset(fileModelKey):
    for model in range(10):
        dataset = []
        print("Start Shadow Model " + str(model))
        with open(fileModelKey, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                label = row[0]
                filename = row[1]
                key = row[2]
                if key[model] == "1":
                    with Image.open(filename) as im:
                        pixels = list(im.convert("L").resize((64,64)).getdata())
                        row = [label] + pixels
                        dataset.append(row)
        with open("shadowTrainingSet_" + str(model) + ".csv", "w", newline='') as outputfile:
            writer = csv.writer(outputfile)
            writer.writerows(dataset)


if __name__ == '__main__':
    # fileKey("artbench-10-imagefolder-split/train/",10)
    createDataset("fileModelKey.csv")
