# Importing necessary python modules
import kagglehub
import os
import shutil

class_help = '''******************** Help on the Class ImportData() *************************
CLASS NAME: ImportData()  
PURPOSE: - This class is to import dataset from kaggle and download it to desired path 
********************************* End of Help  ************************************\n'''
class ImportData():

    # class Init function
    def __init__(self):
         print(class_help)
         pass
 
    # Download the dataset from kaggle
    def download(self, dest_dir):              
        if os.path.exists(dest_dir) and os.path.isdir(dest_dir):
                shutil.rmtree(dest_dir)

        # Download latest version
        download_path = kagglehub.dataset_download("nunenuh/flickr8k")

        print("Path to dataset files:", download_path)

        # Check if it's a file (not a directory)
        if not os.path.isfile(download_path):
            shutil.copytree(download_path, dest_dir)
            print("Files and folders copied successfully.")
