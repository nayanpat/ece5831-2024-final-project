# Project Title: Image Caption Generator Using CNN and LSTM
## Introduction
In this project, a neural network architecture has been proposed and developed using convolutional neural networks (CNN) and long short-term memory (LSTM) layers.  A step by step guide has been provided in this report on how the Image Caption Generator model has been developed and what performance measures were observed.

The project has several code files and massive dataset that includes images, text file, data objects and models.  Following are the details of each of those artifacts.

### Github repo https://github.com/nayanpat/ece5831-2024-final-project
### File Structure:
1. feature_extractor.py - This file has a class that is used to extract feature vector from the image dataset
2. import_dataset.py - This is to import flickr8k dataset using Keras library 
3. caption_data_processing.py - This has a class to process caption data
4. data_generator.py - This is to generate training and validation dataset and generator for fit() function
5. caption_generator_architecture.py - This has a class to define the model architecture and function to train, save and load model
6. model_prediction.py -  This is to do prediction using generated model
7. caption_generator.py - This is a script that is used to select and display an image alongwith its caption
8. final-project.ipynb - This IPython notebook is to show how to run the project and it hs executed cells with all required class

### Link to other project artifacts
##### dataset - https://drive.google.com/file/d/1uy-U1Dbr01SsqyEyqvIvk_vB-KcB3_tJ/view?usp=drive_link
##### Presentation - https://docs.google.com/presentation/d/1MhNExQIUH_PRmOCTh5AUynyTn9ReAVNu/edit?usp=drive_link&ouid=115485934865010381115&rtpof=true&sd=true
##### Presentation Video - https://youtu.be/OfCuIIHYjoQ
##### Report - https://drive.google.com/file/d/1AwVGq0DuwWk_Hgu_2LXfhWfUJqee-bcT/view?usp=drive_link
##### Demo video - https://youtu.be/SMtimpTQ2TM

### User's Manual
1. Download the dataset.zip file using the link above.  
2. The dataset.zip file contains following folders, so extract those folders directly under the folder where all the python code, script and IPython notebook files are located.
    1) dataset - This folder has all 8092 images under a folder 'images'.  This folder also contains the 'captions.txt' file which has human developed five captions for each image.
    2) generated_data - This folder contains all the pickle files needed for loading important data objects, such as features, tokenizer, max caption length, vocabulary size etc. which are necessary for model training and prediction.  Pickle module's load() function is used to load the contents of those pickle files.
    3) models - It has all the generated model which can be used to make prediction.  Keras load_model() function is used to load the model object.
3. Now, you are ready to deploy the script that is used to test your image for generating caption.  For doing that, run following commands from the terminal:

"conda activate ece5831-2024"  
"python caption_generator.py xyz"    
The xyz here is the command line argument for the name of the feature extraction algorithm that you want to try out.  For VGG16, type VGG16 and for Xception, type Xception.  
4. The above command will lead to a launching of a GUI that looks like following:

![alt text](image.png)

5. Click on the "Choose Image" button that opens a pop up window which allows you to select an image from the image database.  
6.  Once selected the image file, click on "Classify Image" button that pops up the image on a separate window.  Close that window and click on the "Classify Image" button again that will run the prediction function in the backend and returns the caption that's displayed on the main GUI window.