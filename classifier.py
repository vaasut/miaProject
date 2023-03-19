import csv
from PIL import Image
from sklearn.linear_model import RidgeClassifier
from joblib import dump
from joblib import load

def trainModel(trainingData):
    with open(trainingData, newline='') as csvfile:
        labels = []
        trainingSet = []
        reader = csv.reader(csvfile)
        count = 0
        for row in reader:
            if (count % 100) == 0:
                labels.append(int(row[0]))
                values = []
                for i in range(len(row)-1):
                    values.append(int(row[i+1]))
                trainingSet.append(values)
            count += 1
        print(trainingSet)
        print(labels)
        classifierModel = RidgeClassifier().fit(trainingSet,labels)
        return classifierModel

def testModel(model, filename):
    with Image.open(filename) as im:
        pixels = list(im.convert("P").resize((64,64)).getdata())
        prediction = model.predict([pixels])
        print(model.decision_function([pixels]))
    return prediction

def saveModel(model, filename):
    dump(model, filename)

def loadModel(modelFile):
    model = load(modelFile)
    return model

filename = "artbench-10-imagefolder-split/test/post_impressionism/anita-malfatti_pedras-na-praia.jpg"
model = trainModel("shadowTrainingPaletteSet_0.csv")
saveModel(model,"myModel.joblib")
model = loadModel("myModel.joblib")
prediction = testModel(model, filename)
print(prediction)