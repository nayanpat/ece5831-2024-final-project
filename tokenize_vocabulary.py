from tensorflow.keras.preprocessing.text import Tokenizer
from pickle import dump

class TokenVocab():

    # Init function for the class
    def __init__(self, descriptions):
        self.desc_list = []
        self.descriptions = descriptions
        self.desc_list = self._dict_to_list()
        pass

    # Convert training descriptions from a dictionary to list
    def _dict_to_list(self):
        desc_list = []
        for key in self.descriptions.keys():
            [desc_list.append(d) for d in self.descriptions[key]]
        return desc_list

    # Creating tokenizer class. This will vectorise text corpus each integer 
    # will represent token in dictionary
    def create_tokenizer(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.desc_list)
        return tokenizer
    
    # Save the tockens in a pickel file
    def save_token(self, tokenizer, filename):
        with open(filename, 'wb') as file:
            dump(tokenizer, file)

    # Calculate maximum length of descriptions
    def max_length(self):
        return max(len(d.split()) for d in self.desc_list)
        
    