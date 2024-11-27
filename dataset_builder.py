# Importing necessary modules
import os

class DatasetBuilder():

    # Class init function
    def __init__(self):
        self.img_file_names = []
        self.train_dataset_file_names = []
        self.test_dataset_file_names = []
        self.validation_dataset_file_names = []

    # Create a list of image file names
    def file_name_extractor(self, image_dir):
        for entry in os.scandir(image_dir):
            if entry.is_file():
                self.img_file_names.append(entry.name)
        
    # Split image files into train, test and validation dataset
    def split_dataset(self):
        self.train_dataset_file_names = self.img_file_names[0:6000]
        self.validation_dataset_file_names = self.img_file_names[6000:7000]
        self.test_dataset_file_names = self.img_file_names[7000:]
        print(self.test_dataset_file_names)

    def _create_file(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
            print(f"File '{filename}' deleted successfully.")
        else:
            print(f"File '{filename}' does not exist.")

        return open(filename, 'w')

    # Save train, test and validation file names into text files
    def save_files(self, file_path):        
        train_dataset_file = file_path + '/' + 'train_dataset.txt'
        validation_data_file = file_path + '/' + 'validation_dataset.txt'
        test_data_file = file_path + '/' + 'test_dataset.txt'

        # Writing training file names to a file        
        file = self._create_file(train_dataset_file)
        file.writelines(line + '\n' for line in self.train_dataset_file_names)
        file.close()

        # Writing validation file names to a file        
        file = self._create_file(validation_data_file)
        file.writelines(line + '\n' for line in self.validation_dataset_file_names)
        file.close()

        # Writing test file names to a file        
        file = self._create_file(test_data_file)
        file.writelines(line + '\n' for line in self.test_dataset_file_names)
        file.close()
