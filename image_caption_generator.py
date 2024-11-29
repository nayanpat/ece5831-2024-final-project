import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from pickle import load
from tensorflow.keras.applications.xception import Xception

class ImageCaptionGenerator():

    # Class init function
    def __init__(self):  
        # Load tokenizer pickle file to get tokenizer and max vocabulary length
        with open(os.path.join(os.getcwd(), 'models', 'tokenizer.pkl'), 'rb') as file:
            token_data = load(file)
        self.tokenizer = token_data[0]
        self.max_length = token_data[1]

        # Load the trained model
        model_file = os.path.join(os.getcwd(), 'models', 'model.patel_caption_generator_cnn.keras')
        self.model = load_model(model_file)
 
    def extract_features(self, img_file_name):
            # Use xception model for feature extraction
            xception_model = Xception(include_top=False, pooling="avg")
            try:
                image = Image.open(img_file_name)
            except:
                print("ERROR: Wrong Image File or Path.  Check and Try Again!")
            image = image.resize((224, 224))
            image = np.array(image)
            
            # for images that has 4 channels, we convert them into 3 channels
            if image.shape[2] == 4: 
                image = image[..., :3]
            image = np.expand_dims(image, axis=0)

            # Normalize the image
            image = image / 127.5
            image = image - 1.0
            features = xception_model.predict(image)

            return features

    def word_for_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
            
        return None

    def generate_desc(self, img_features):
        in_text = 'start'
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            pred = self.model.predict([img_features, sequence], verbose=0)
            pred = np.argmax(pred)
            word = self.word_for_id(pred, self.tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
            
        return in_text
