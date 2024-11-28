# Importing Necessary Python Module
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from model_data_generator import ModelDataGenerate


class_help = '''******************** Help on the Class InputDataProcessing() *************************
CLASS NAME: InputDataProcessing()  
PURPOSE: - This class is to perform the input image and caption text data processing
INPUT PARAMETERS:  This class needs following input parameters when instantiating
                   1)  batch_size - Training batch size
                   2)  epochs -  Number of epochs
MEMBER FUNCTIONS:  
1) __init__():  This is for class initization
2) _load_data():  This is for opening the caption text file and reading texts from it
PUBLIC FUNCTIONS:
1) image_description(): This function reads the texts from caption text file and 
                        create dictionary of image file names and descriptions associated with them  
2) clean_data():  This function cleans text data - convert all text to lower case, remove punctuation, 
                  remove single text
3) text_vocabulary(): This function separates all the unique texts and create a vocabulary list from 
                      all the caption descriptions
4) save_descriptions():  This function saves image caption descriptions in a file 
********************************* End of Help  ************************************\n'''

class CaptionGeneratorModelArchitecture():

    # Class init function
    def __init__(self, 
                 train_descriptions,
                 valid_descriptions,
                 train_features,
                 valid_features,
                 tokenizer,
                 desc_list,
                 vocab_size, 
                 max_length, 
                 epochs,
                 batch_size, 
                 learn_rate):
        self.train_descriptions = train_descriptions
        self.valid_descriptions = valid_descriptions
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.train_features = train_features
        self.valid_features = valid_features
        self.tokenizer = tokenizer
        self.desc_list = desc_list
        self.epochs = epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate

    # Define the captioning model
    def design_model(self):

        # Features from the CNN model squeezed from 2048 to 256 nodes
        Input1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(Input1)
        fe2 = Dense(256, activation='relu')(fe1)

        # LSTM sequence model
        Input2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(Input2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256, unroll=True)(se2)

        # Merging both models
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

        # tie it together [image, seq] [word]
        model = Model(inputs=[Input1, Input2], outputs=outputs)

        # Compile the model
        Optimizer = optimizers.Adam(learning_rate=self.learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=['accuracy'])

        return model
    
    # train our model
    def train_model(self):
        # Create a list to store the history objects
        history_log = []
        # Instantiate the ModelDataGenerate() Class for train dataset generation
        TrainDataGenerate = ModelDataGenerate(self.train_descriptions, 
                                              self.train_features, 
                                              self.tokenizer, 
                                              self.max_length,
                                              self.desc_list,
                                              self.vocab_size)
        
        # Instantiate the ModelDataGenerate() Class for valid dataset generation
        ValidDataGenerate = ModelDataGenerate(self.valid_descriptions, 
                                              self.valid_features, 
                                              self.tokenizer, 
                                              self.max_length,
                                              self.desc_list,
                                              self.vocab_size)

        callbacks = [tf.keras.callbacks.EarlyStopping(patience=20),
                    tf.keras.callbacks.TensorBoard(log_dir='./logs')]
        
        model = self.design_model()
        steps = len(self.train_descriptions) // self.batch_size
        validation_steps = len(self.valid_descriptions) // self.batch_size
        # steps = len(self.train_descriptions)
        # Train the model for no. of epochs
            # train_data = TrainDataGenerate.data_generator()
            # valid_data = ValidDataGenerate.data_generator()
        for i in range(self.epochs):
            print(f'Running Epoch # {i+1} of {self.epochs}.............................')
            train_data = TrainDataGenerate.data_generator()
            valid_data = ValidDataGenerate.data_generator()
            history = model.fit(train_data, 
                                validation_data=valid_data,
                                epochs=1, 
                                batch_size=self.batch_size, 
                                steps_per_epoch=steps, 
                                callbacks=callbacks,
                                validation_steps=validation_steps,
                                verbose=1)
            # Append the history object to the list
            history_log.append(history.history)
        
        return model, history_log
    
    # Save the model
    def save_model(self, model, model_file):
        model.save(model_file)
    
    # Load the keras model
    def load_model(self, model_file_name):
        model = load_model(model_file_name)
    
        # Summarize model
        print(model.summary())
        plot_model(model, to_file='model.png', show_shapes=True)

        return model
    
    # Predict input(s)
    def predict(self, model_file_name, images):
        return model_file_name.predict(images)
