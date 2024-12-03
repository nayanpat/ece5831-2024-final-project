from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class ModelPrediction():

    # class Init function
    def __init__(self, model):
        self.model = model
 
    # A untility function for getting a word with an index
    def get_word_from_index(sel, index, tokenizer):
        return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)
    
    # Function to return predicted caption
    def predict_caption(self, image_features, tokenizer, max_caption_length):
        # Initialize the caption sequence
        print(f'Image Feature = {image_features}')
        caption = 'start'
        
        # Generate the caption
        for _ in range(max_caption_length):
            # Convert the current caption to a sequence of token indices
            sequence = tokenizer.texts_to_sequences([caption])[0]
            # Pad the sequence to match the maximum caption length
            sequence = pad_sequences([sequence], maxlen=max_caption_length)
            # Predict the next word's probability distribution
            yhat = self.model.predict([image_features, sequence], verbose=0)
            # Get the index with the highest probability
            predicted_index = np.argmax(yhat)
            # Convert the index to a word
            predicted_word = self.get_word_from_index(predicted_index, tokenizer)
            
            # Append the predicted word to the caption
            caption += " " + predicted_word
            
            # Stop if the word is None or if the end sequence tag is encountered
            if predicted_word is None or predicted_word == 'end':
                break
        
        return caption
    