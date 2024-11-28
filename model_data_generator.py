# Importing Necessary Python Modules
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class_help = '''******************** Help on the Class ImportData() *************************
CLASS NAME: ImportData()  
PURPOSE: - This class is to generate data for the model input
MEMBER FUNCTIONS:  
1) __init__():  This is for class initization
PUBLIC FUNCTIONS:
1) download(): This function accesses the flickr8k dataset images and caption text files
                and download that to the dest_dir folder.  This function needs destination
                path as an input parameter
********************************* End of Help  ************************************\n'''

class ModelDataGenerate():

    # Init function for the class
    def __init__(self, 
                 descriptions, 
                 features, 
                 tokenizer, 
                 max_length,
                 desc_list,
                 vocab_size):
        self.descriptions = descriptions
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.desc_list = desc_list
        self.vocab_size = vocab_size

    # Create sequence in input and output data
    def _create_sequences(self, feature, description_list):
        X1, X2, y = list(), list(), list()
        # walk through each description for the image
        for desc in description_list:
            # encode the sequence
            seq = self.tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                # store
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)

    # Generate data to be used by model.fit_generator()
    def data_generator(self):
        while 1:
            for key, description_list in self.descriptions.items():
                # Retrieve image features
                feature = self.features[key][0]
                input_image, input_sequence, output_word = self._create_sequences(feature, description_list)
                yield (input_image, input_sequence), output_word
