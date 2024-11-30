# Import Necessary Python Modules
import string
import numpy as np
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tqdm.notebook import tqdm
tqdm().pandas()
from pickle import dump, load
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ExtractFeatures():

    # class Init function
    def __init__(self, feature_extraction_model):
        self.model = None
        self.feature_extraction_model = feature_extraction_model
    
    def init_model(self):

        try: 
            if self.feature_extraction_model == 'Xception':
                model = Xception(include_top=False, pooling='avg')
            if self.feature_extraction_model == 'VGG16':
                model = VGG16()
                # Restructuring the model to remove the last classification layer, this will give us access to the output features of the model
                model = Model(inputs=model.inputs, outputs=model.layers[-2].output) 
        except:
            print("ERROR:  Wrong Feature Extraction Model Name Entered.")

        self.model = model

    def extract_features(self, image_file_path):
        # Initialize an empty dictionary to store image features
        image_features = {}
        
        # Loop through each image in the directory
        for img_name in tqdm(os.listdir(image_file_path)):
            # Load the image from file
            img_path = os.path.join(image_file_path, img_name)
            image = load_img(img_path, target_size=(224, 224))
            # Convert image pixels to a numpy array
            image = img_to_array(image)
            # Reshape the data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

            # Pre-process the input images
            if self.feature_extraction_model == 'Xception':
                image = tf.keras.applications.xception.preprocess_input(image)
            if self.feature_extraction_model == 'VGG16':
                image = tf.keras.applications.vgg16.preprocess_input(image)
            # Preprocess the image for ResNet50
            
            # Extract features using the pre-trained model
            image_feature = self.model.predict(image, verbose=0)
            # Get the image ID by removing the file extension
            image_id = img_name.split('.')[0]
            # Store the extracted feature in the dictionary with the image ID as the key
            image_features[image_id] = image_feature

        return image_feature
    
    # Save image features to a pickle file
    def save_features(self, image_feature, image_feature_file):
        # Save feature vector
        with open(image_feature_file, 'wb') as file:
            dump(image_feature, file)

    # Load feature model 
    def load_feature_model(self, image_feature_file):
        with open(image_feature_file, 'rb') as file:
            return load(file)