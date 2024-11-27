# Import data_processing module
from data_processing import InputDataProcessing
from pickle import load

class LoadTrainDataset():

    # Init function for the class
    def __init__(self):
        self.DataProcessing = InputDataProcessing() 
        self.images = None       

    # load the image data 
    def load_images(self, filename):
        data = self.DataProcessing.load_data(filename)
        self.images = data.split('\n')[:-1]
        return self.images
    
    # Load Cleaned Descriptions
    def load_clean_descriptions(self, filename): 
        file = self.DataProcessing.load_data(filename)
        descriptions = {}
        for line in file.split('\n'):
            texts = line.split()
            if len(texts) < 1 :
                continue
            image, image_caption = texts[0], texts[1:]
            if image in self.images:
                if image not in descriptions:
                    descriptions[image] = []
                desc = '<start> ' + " ".join(image_caption) + ' <end>'
                descriptions[image].append(desc)
        return descriptions
    
    # Load features from the features.pkl file and selecting only needed features
    def load_features(self, images):
        all_features = load(open("features.pkl","rb"))
        features = {idx:all_features[idx] for idx in images}
        return features