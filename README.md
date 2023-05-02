# miaProject
This is the code for my machine learning project about Membership Inference Attacks.

## Notes about Running Code
It's important to note that none of the training or test images are in this directory. In addition, none of the stored models I made are uploaded here. 

In order to run this code, you will need to download the images from artbench and create the proper directories. I also ran code to resize the images so that they were smaller. Once you do this, you can then train the models, and then load the models to test.

### Python Packages You Will Need
- csv
- joblib
- numpy
- math
- PIL
- pyplot
- os
- random
- sklearn
- tensorflow
- time

## Explanation of Python Files in Alphabetical Order.

<b> colorConversion.py </b> - This file has different functions to encode image colors in another color space. I was using this for some of my attempts to make the shadow models as I thought it would help to compress the images while keeping still relevant color information.

<b> EvaluateTensorClassifier.py </b> - This file has code to evaluate the prediction correctness and prediction confidence of images that belong (or don't belong) to different tensor classifier's training sets. For each of the different experiments I ran, I modified the functions slightly to fit them and get the results I needed.

<b> fileModelKey.py </b> - This file has functions to create a file model key. This key would work by assigning each image a bit string corresponding to the shadow models that would be using it in their training sets. For example a string of 1011110000 would mean the first, third, fourth, fifth and sixth shadow models would be using that image and the other models would not.

<b> generateColorImage.py </b> - This contains code for me trying to create my own GAN shadow models using the tensor classifier. This method did not work well, as I explain in my report.

<b> generateImage.py </b> - This contains code for met trying to create my own GAN shadow model. This method did not work well, as I explain in my report.

<b> ridgeClassifier.py </b> - This contains code for a ridge classifier model. The ridege classifier was overfitting and not working as well as the tensorclassifier so I swapped to using the tensor classifier.

<b> tensorClassifier.py </b> - This contains code for the tensor classifier. This code is from the tensorflow classifier tutorial.

<b> tensorGan.py </b> - This contains code for the tensor GAN. This code is from the tensorflow GAN tutorial

