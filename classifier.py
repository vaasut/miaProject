import csv
from PIL import Image
from sklearn import svm

def trainModel(trainingData):
    with open(trainingData, newline='') as csvfile:
        labels = []
        trainingSet = []
        reader = csv.reader(csvfile)
        count = 0
        for row in reader:
            if (count % 10) == 0:
                labels.append(int(row[0]))
                values = []
                for i in range(len(row)-1):
                    values.append(int(row[i+1]))
                trainingSet.append(values)
            count += 1
        print(trainingSet)
        print(labels)
        clfSVM = svm.LinearSVC(C=0.1).fit(trainingSet,labels)
        return clfSVM

def testModel(model,filename):
    with Image.open(filename) as im:
        pixels = list(im.convert("P").resize((64,64)).getdata())
        prediction = model.predict([pixels])
    return prediction

filename = "artbench-10-imagefolder-split/test/baroque/abraham-storck_italianate-park-landscape.jpg"
model = trainModel("shadowTrainingPaletteSet_0.csv")
prediction = testModel(model,filename)
print(prediction)