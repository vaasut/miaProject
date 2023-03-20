from PIL import Image
from joblib import load
import random
from sklearn.linear_model import RidgeClassifier
import colorConversion
import tensorClassifier

def generateColorSeedImage(colorKey):
    imagePixels = []
    for pixel in range(16384): #for 128 x 128 image
        color = colorKey[random.randint(0,864)]
        imagePixels.append((color[0],color[1],color[2]))
    return imagePixels

def randomChangeColorPixels(imagePixels,numChanges,colorKey):
    colors = len(colorKey)
    pixelsToChange = set()
    changedPixels = []
    for i in range(numChanges):
        pixelsToChange.add(random.randint(0,16383))
    for i in range(16384):
        if i in pixelsToChange:
            choice = random.randint(0,4)
            if choice == 0:
                changedPixels.append(imagePixels[(i+1)%4096]) #pixel adjacent
            elif choice == 1:
                changedPixels.append(imagePixels[(i + 64)%4096])  # pixel under
            elif choice == 2:
                changedPixels.append(imagePixels[(i - 1) % 4096])  # pixel left
            elif choice == 3:
                changedPixels.append(imagePixels[(i - 64) % 4096])  # pixel above
            else:
                changedPixels.append(colorKey[random.randint(0,colors-1)])
        else:
            changedPixels.append(imagePixels[i])
    return changedPixels


def displayColorImage(imagePixels):
    image_out = Image.new(mode="RGB", size=(128, 128))
    image_out.putdata(imagePixels)
    image_out.show()


def colorCycle(imagePixels,numIterations,numChanges,colorKey):
    for i in range(numIterations):
        changedPixels = randomChangeColorPixels(imagePixels, numChanges,colorKey)
    return changedPixels

def getTensorFormat(imagePixels):
    tensorPixels = []
    for row in range(128):
        rowPixels = []
        for column in range(128):
            pixel = row*128+column
            rowPixels.append([imagePixels[pixel][0],imagePixels[pixel][1],imagePixels[pixel][2]])
        tensorPixels.append(rowPixels)
    return tensorPixels
def generateColorImage(modelFile,label,branches,numIterations):
    colorKey = colorConversion.getColorKey("colorKey.csv")
    model = tensorClassifier.loadModel()
    imagePixels = generateColorSeedImage(colorKey)
    displayColorImage(imagePixels)
    imagePixels = colorCycle(imagePixels, 10000, 100,colorKey)
    # print(len(imagePixels))
    tensorPixels = getTensorFormat(imagePixels)
    for i in range(numIterations):
        branchScore = -10
        for j in range(branches):
            temp = colorCycle(imagePixels, 10000, 100, colorKey)
            tensorPixels = getTensorFormat(imagePixels)
            tempScore = tensorClassifier.getScore(model, tensorPixels)[label]
            if (tempScore > branchScore):
                imagePixels = temp
                branchScore = tempScore
        print(branchScore)
        if (i % 5) == 0:
            print("Iteration " + str(i))
            displayColorImage(imagePixels)
    displayColorImage(imagePixels)

generateColorImage("saved_model/myTensorFlowModel",1,10,10)