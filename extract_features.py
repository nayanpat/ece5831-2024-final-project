# Import Necessary Python Modules
import string
import numpy as np
from PIL import Image
import os
import numpy as np
from tensorflow.keras.applications.xception import Xception
from tqdm.notebook import tqdm
tqdm().pandas()
from pickle import dump, load

class ExtractFeatures():

    # class Init function
    def __init__(self, ):
         pass
    
    # Extract the feature vector from all images from dataset
    def extract_features(self, img_dir, kera_model):
        if kera_model == 'Xception':
            model = Xception(include_top=False, pooling='avg')
        features = {}
        # Loop thru each image file, resize/normalize the image and run
        # prediction to extract feature list
        for img in tqdm(os.listdir(img_dir)):
            filename = img_dir + '/' + img
            image = Image.open(filename)
            image = image.resize((224, 224)) # Resize image
            image = np.expand_dims(image, axis=0)
            # Normalize image data
            image = image / 127.5
            image = image - 1.0
            # Get predicted features and save in the dictionary
            feature = model.predict(image)
            features[img] = feature
        return features
    
    # Save features in the pickel file
    def save_feature_model(self, features, feature_model_name):
        # Save features 2048 feature vector
        dump(features, open(feature_model_name, "wb"))

    # Load feature model 
    def load_feature_model(self, feature_model_name):
        return load(feature_model_name)
