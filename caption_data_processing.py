# Import Necessary Python Modules
import string
import numpy as np
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm
tqdm().pandas()
from pickle import dump, load
from collections import defaultdict
from tensorflow.keras.preprocessing.text import Tokenizer

class CaptionDataProcessing():

    # class Init function
    def __init__(self):
        pass

    # Read the caption text file
    def _read_caption_file(self, caption_text_file):
        with open(caption_text_file, 'r') as file:
            next(file) # Skip reading first header line
            file_obj = file.read()
            file.close()
        return file_obj

    def image_to_caption_mapping(self, caption_text_file):
        # Object of caption read file 
        caption_file = self._read_caption_file(caption_text_file)

        # Create mapping of image to captions
        image_to_captions_mapping = defaultdict(list)

        # Process lines from self.caption_file
        for line in tqdm(caption_file.split('\n')):
            # Split the line by comma(,)
            tokens = line.split('|')
            if len(tokens) < 2:
                continue
            image_id, *captions = tokens[0], tokens[2]
            # Remove extension from image ID
            image_id = image_id.split('.')[0]
            # Convert captions list to string
            caption = " ".join(captions)
            # Store the caption using defaultdict
            image_to_captions_mapping[image_id].append(caption)    

        return image_to_captions_mapping
    
    # Function for processing the captions
    def clean_data(self, image_to_captions_mapping):
        image_to_captions_mapping_table = {}
        for key, captions in image_to_captions_mapping.items():
            for idx in range(len(captions)):
                # Take one caption at a time
                caption = captions[idx]
                # Preprocessing steps
                # Convert to lowercase
                caption = caption.lower()
                # Remove non-alphabetical characters
                caption = ''.join(char for char in caption if char.isalpha() or char.isspace())
                # Remove extra spaces
                caption = caption.replace(r'\s+', ' ')
                # Add unique start and end tokens to the caption
                caption = '<start> ' + ' '.join([word for word in caption.split() if len(word) > 1]) + ' <end>'
                captions[idx] = caption
            image_to_captions_mapping_table[key] = captions
        return image_to_captions_mapping_table
    
    # Function to create tokenizer from all the caption texts
    def tokenize_caption(self, all_captions):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)

        return tokenizer

    # Save the tokenizer
    def save_tokenizer(self, tokenizer, tokenizer_pickle_file):        
        with open(tokenizer_pickle_file, 'wb') as tokenizer_file:
            dump(tokenizer, tokenizer_file)

    # Load the tokenizer
    def load_tokenizer(self, tokenizer_pickle_file):
        with open(tokenizer_pickle_file, 'rb') as tokenizer_file:
            return load(tokenizer_file)

    # Calculate max caption length
    def max_caption_length(self, tokenizer, all_captions):
        max_caption_length = max(len(tokenizer.texts_to_sequences([caption])[0]) for caption in all_captions)
        return max_caption_length
    
    # Calculate total vocabulary size 
    def vocabulary_size(self, tokenizer):
        vocab_size = len(tokenizer.word_index) + 1
        return vocab_size