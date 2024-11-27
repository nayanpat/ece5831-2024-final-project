# Import Necessary Python Modules
import string
import re

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

class InputDataProcessing():

    # Init function for the class
    def __init__(self):
        pass

    # Load the caption text file
    def load_data(self, filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()
        return text
 
    # Read the texts from caption text file and 
    # create dictionary of image file names and 
    # descriptions associated with them
    def image_description(self, texts):
        lines = texts.split('\n')
        lines.pop(0)  # Delete the header as that's not needed
        img_caption = {}
        for line in lines[:-1]:
            img, line = re.split(r'\|.*?\|', line)
            if img[:-2] not in line:
                img_caption[img[:]] = [line]
            else:
                img_caption[img[:]].append(line)
        return img_caption
    
    #  Clean text data - convert all text to lower case, remove punctuation,
    #  remove single text
    def clean_data(self, image_captions):
        # Convert the punctuations to unicode to process later
        table = str.maketrans('', '', string.punctuation)
        cleaned_image_captions = image_captions
        for img, caps in image_captions.items():
            for idx, img_caption in enumerate(caps):
                # Remove '-' from the texts
                img_caption.replace("-"," ")
                description = img_caption.split()

                # Convert all texts to lower case
                description = [word.lower() for word in description]

                # Remove punctuation from each string
                description = [word.translate(table) for word in description]

                # Remove hanging 's and a/A (word with a single alphabet)
                description = [word for word in description if(len(word)>1)]
                 
                # Remove texts with numbers in them
                description = [word for word in description if(word.isalpha())]

                # Convert individual texts back to string
                img_caption = ' '.join(description)
                cleaned_image_captions[img][idx] = img_caption
                
        return cleaned_image_captions
    
    # Separate all the unique texts and create a vocabulary list from all the caption descriptions
    def text_vocabulary(self, caption_descriptions):
        # create an empty set of vocabulary
        vocabulary = set()
        
        for key in caption_descriptions.keys():
            [vocabulary.update(d.split()) for d in caption_descriptions[key]]
        
        return vocabulary
    
    # Save image caption descriptions in a file 
    def save_descriptions(self, caption_descriptions, filename):
        lines = list()
        for key, desc_list in caption_descriptions.items():
            for desc in desc_list:
                lines.append(key + '\t' + desc )
        data = "\n".join(lines)
        file = open(filename, "w")
        file.write(data)
        file.close()