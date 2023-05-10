# miaProject
This is the code for my machine learning project about Membership Inference Attacks.

## Notes about Running Code
It's important to note that none of the training or test images are in this github repository. In order to run the code, please download the two zip files from the project directory in google drive.
- Modified Artbench Dataset (2-Classes with some images removed): https://drive.google.com/file/d/1jyXkkE9wGJ0l1rWL60zkFiy5352jWxmQ/view?usp=sharing
- Art from Specific Artists: https://drive.google.com/file/d/1TWGnnYKDiMnzw18KbnLy-nGCCLpMA68U/view?usp=sharing

When you're testing my code, I would recommend looking most closely at the code in EvaluateTensorClassifier.py and in createShadowModels.py. These two files contain the important code that I wrote for Parts 2 and 3 of my project, relating to prediction-correctness, prediction-confidence, and shadow training to attack a binary classifier. 

The files, colorConversion.py, fileModelKey.py, generateColorImage.py, and generateImage.py have functions relating to trying to train shadow models to work for a GAN. The files RidgeClassifier.py, tensorClassifier.py, and tensorGan.py contain code based on tutorials built in functions.

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

<b> createShadowModels.py </b> - This file contains the code to create the shadow models I used in Part Three of my project. It contains functions to create the datasets for each shadow model and to train each shadow model. It also has code to create the attack model's training set from these shadow models. Finally, it has code to train and evaluate the attack model on a given target model.

<b> EvaluateTensorClassifier.py </b> - This file has code to evaluate the prediction correctness and prediction confidence of images that belong (or don't belong) to different tensor classifier's training sets. For each of the different experiments I ran, I modified the functions slightly to fit them and get the results I needed.

<b> fileModelKey.py </b> - This file has functions to create a file model key. This key would work by assigning each image a bit string corresponding to the shadow models that would be using it in their training sets. For example a string of 1011110000 would mean the first, third, fourth, fifth and sixth shadow models would be using that image and the other models would not.

<b> generateColorImage.py </b> - This contains code for me trying to create my own GAN shadow models using the tensor classifier. This method did not work well, as I explain in my report.

<b> generateImage.py </b> - This contains code for met trying to create my own GAN shadow model. This method did not work well, as I explain in my report.

<b> ridgeClassifier.py </b> - This contains code for a ridge classifier model. The ridege classifier was overfitting and not working as well as the tensorclassifier so I swapped to using the tensor classifier.

<b> tensorClassifier.py </b> - This contains code for the tensor classifier. This code is from the tensorflow classifier tutorial.

<b> tensorGan.py </b> - This contains code for the tensor GAN. This code is from the tensorflow GAN tutorial

