# Importing necessary python modules
import kagglehub
import os
import shutil

class_help = '''******************** Help on the Class ImportData() *************************
CLASS NAME: ImportData()  
PURPOSE: - This class is to import dataset from kaggle and download it to desired path 
MEMBER FUNCTIONS:  
1) __init__():  This is for class initization
PUBLIC FUNCTIONS:
1) download(): This function accesses the flickr8k dataset images and caption text files
                and download that to the dest_dir folder.  This function needs destination
                path as an input parameter
********************************* End of Help  ************************************\n'''

class ImportData():

    # class Init function
    def __init__(self):
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
