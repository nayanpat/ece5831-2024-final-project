# Importing Necessary Python Module
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, concatenate, \
                                    RepeatVector, Bidirectional, Dot, Lambda, Activation
from model_training_data_prep import ModelTrainDataPrep

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

class ModelArchitectureAndTraining():

    # Class init function
    def __init__(self, train_data, validation_data, image_to_captions_mapping, loaded_features,
                 tokenizer, max_caption_length, vocab_size, epochs=70, batch_size=32, learning_rate=0.001):

        self.train_data = train_data[0]
        self.validation_data = validation_data[0]
        self.image_to_captions_mapping = image_to_captions_mapping
        self.loaded_features = loaded_features
        self.tokenizer = tokenizer

        self.epochs = epochs
        self.batch_size = batch_size
        self.max_caption_length = max_caption_length
        self.vocab_size = vocab_size
        self.learn_rate = learning_rate

    # Define the captioning model
    def design_model(self, feature_extraction_model):

        try:
            if feature_extraction_model == 'Xception':
                input_shape = 2048
            if feature_extraction_model == 'VGG16':
                input_shape = 4096
        except:
            print('ERROR: Wrong Feature Extraction Model Name Entered.')              

        # Encoder model
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        fe2_projected = RepeatVector(self.max_caption_length)(fe2)
        fe2_projected = Bidirectional(LSTM(256, return_sequences=True))(fe2_projected)

        # Sequence feature layers
        inputs2 = Input(shape=(self.max_caption_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = Bidirectional(LSTM(256, unroll=True, return_sequences=True))(se2)

        # Apply attention mechanism using Dot product
        attention = Dot(axes=[2, 2])([fe2_projected, se3])  # Calculate attention scores

        # Softmax attention scores
        attention_scores = Activation('softmax')(attention)

        # Apply attention scores to sequence embeddings
        attention_context = Lambda(lambda x: tf.einsum('ijk,ijl->ikl', x[0], x[1]))([attention_scores, se3])

        # Sum the attended sequence embeddings along the time axis
        # context_vector = tf.reduce_sum(attention_context, axis=1)
        ReduceSum_Obj = ReduceSum()
        context_vector = ReduceSum_Obj(attention_context)

        # Decoder model
        decoder_input = concatenate([context_vector, fe2], axis=-1)
        decoder1 = Dense(256, activation='relu')(decoder_input)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder1)

        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        # Compile the model
        Optimizer = optimizers.Adam(learning_rate=self.learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=['accuracy'])

        return model
    
    # train our model
    def train_model(self, model):
        # Calculate the steps_per_epoch based on the number of batches in one epoch
        steps_per_epoch = len(self.train_data) // self.atch_size
        validation_steps = len(self.validation_data) // self.batch_size  # Calculate the steps for validation data

        # Instantiate the ModelTrainDataPrep() Class for training data generation
        TrainDataPrep_Obj = ModelTrainDataPrep()

        # Instantiate the ModelTrainDataPrep() Class for training data generation
        ValidDataPrep_Obj = ModelTrainDataPrep()

        history_log = []
        # Loop through the epochs for training
        for epoch in range(self.epochs):
            print(f'Running Epoch {self.epoch+1} of {self.epochs}.............')
            
            # Set up data generators
            train_generator = TrainDataPrep_Obj.data_generator(self.train_data, self.image_to_captions_mapping, 
                                                               self.loaded_features, self.tokenizer, self.max_caption_length, 
                                                               self.vocab_size, self.batch_size)
            test_generator = ValidDataPrep_Obj.data_generator(self.validation_data, self.image_to_captions_mapping, 
                                                              self.loaded_features, self.tokenizer, self.max_caption_length, 
                                                              self.vocab_size, self.batch_size)
            
            history = model.fit(train_generator, epochs=1, steps_per_epoch=steps_per_epoch,
                validation_data=test_generator, validation_steps=validation_steps,
                verbose=1)
            
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

# A wrapper class for avoiding error for reduce_sum() function
class ReduceSum(tf.keras.layers.Layer):
    def __init__(self):
        super(ReduceSum, self).__init__()
   
    def call(self, input):
        return tf.reduce_sum(input, axis=1)
