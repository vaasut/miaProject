from PIL import Image
from joblib import load
import random
from sklearn.linear_model import RidgeClassifier
import colorConversion
import tensorClassifier

def generateSeedImage():
    imagePixels = []
    for i in range(4096): #4096 pixels in image
        imagePixels.append(random.randint(0,255))
    return imagePixels

def randomChangePixels(imagePixels,numChanges):
    pixelsToChange = set()
    changedPixels = []
    for i in range(numChanges):
        pixelsToChange.add(random.randint(0,4095))
    for i in range(4096):
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
                changedPixels.append(random.randint(0,255))
        else:
            changedPixels.append(imagePixels[i])
    return changedPixels

def displayImage(imagePixels,name):
    image_out = Image.new(mode="L", size=(64, 64))
    image_out.putdata(imagePixels)
    image_out.save(str(name) + "_generated.png")
    image_out.show()

def cycle(imagePixels,numIterations,numChanges):
    for i in range(numIterations):
        changedPixels = randomChangePixels(imagePixels, numChanges)
    return changedPixels

def generateImage(modelFile,label,branches,numIterations): #generates 64x64 pixel image
    model = load(modelFile)
    imagePixels = generateSeedImage()
    imagePixels = cycle(imagePixels,1000,100)
    startScore = model.decision_function([imagePixels])[0][label]
    for i in range(numIterations):
        branchScore = -10
        for j in range(branches):
            temp = cycle(imagePixels,1000,100)
            tempScore = model.decision_function([temp])[0][label]
            if (tempScore > branchScore):
                imagePixels = temp
                branchScore = tempScore
        print(branchScore)
        if (i % 5) == 0:
            print("Iteration " + str(i))
            displayImage(imagePixels,i)
    displayImage(imagePixels,i)

generateImage('myModel.joblib',3,50,15)
# imagePixels = generateSeedImage()
# displayImage(imagePixels)