import csv
import math
import os

from PIL import Image

def getColorDistance(colorOne,colorTwo):
   redDistance = colorTwo[0] - colorOne[0]
   greenDistance = colorTwo[1] - colorOne[1]
   blueDistance = colorTwo[2] - colorOne[2]
   distance = math.sqrt(redDistance**2 + greenDistance**2 + blueDistance**2)
   return distance

def getColorKey(csvfile):
    colorKey = []
    with open(csvfile, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            color = (int(row[3]),int(row[4]),int(row[5]))
            colorKey.append(color)
    return colorKey

def getClosestColor(color,colorKey):
    closestColor = 0
    minDistance = 500
    for i in range(len(colorKey)):
        distance = getColorDistance(color,colorKey[i])
        if distance < minDistance:
            minDistance = distance
            closestColor = i
    # print(closestColor)
    return closestColor

def convertImageColor(filename,colorKey):
    with Image.open(filename) as im:
        pixels = list(im.resize((64, 64)).getdata())
        for i in range(len(pixels)):
            pixels[i] = getClosestColor(pixels[i],colorKey)
        return pixels


def convertBackToImage(pixels,colorKey):
    imagePixels = []
    for i in pixels:
        imagePixels.append(colorKey[i])
    print(imagePixels)
    image_out = Image.new("RGB",(64,64))
    image_out.putdata(imagePixels)
    image_out.show()

# colorKey = getColorKey("colorKey.csv")
# filename = "artbench-10-imagefolder-split/train/art_nouveau/a-y-jackson_algoma-in-november-1935.jpg"
# with Image.open(filename) as im:
#     im.resize((64,64)).show()
#     im.convert("L").resize((64,64)).show()
# pixels = convertImageColor(filename,colorKey)
# convertBackToImage(pixels,colorKey)

def resizeImages(folderName):
    count = 0
    labels = ["art_nouveau", "expressionism", "post_impressionism", "realism", "surrealism", "ukiyo_e"]
    for label in labels:
        print("Starting Resize of " + label)
        files = os.listdir(folderName + label + "/")
        for file in files:
            with Image.open(folderName + label + "/" + file) as im:
                imResize = im.resize((64,64))
                imResize.save(folderName + label + "/" + file, "JPEG",quality=95)
            if (count % 1000) == 0:
                print(count)
            count += 1

# resizeImages("artbench-10-imagefolder-split/test/")