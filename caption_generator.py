# Importing necessary modules
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from model_predicton import ModelPrediction
import os, argparse
from pickle import load
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.xception import Xception 
from tensorflow.keras.applications.vgg16 import VGG16

class CaptionGeneratorGUI():

    # class Init function
    def __init__(self, caption_generator_model, 
                 features, 
                 feature_extraction_algo_name,
                #  image, 
                 frame):
        self.caption_generator_model = caption_generator_model
        self.features = features
        self.feature_extraction_algo_name = feature_extraction_algo_name
        self.image = None
        self.frame = frame

        # Instantiate the ModelPrediction() Class
        self.ModelPrediction_Obj = ModelPrediction(self.caption_generator_model)

    # Function to get image using the GUI
    def get_image(self):
        image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                        filetypes=(("all files", "*.*"), ("JPEG files", "*.jpg")))
        self.image = image_data
        basewidth = 300
        img = Image.open(image_data)
        # wpercent = (basewidth / float(img.size[0]))
        # hsize = int((float(img.size[1]) * float(wpercent)))
        # img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        file_name = image_data.split('/')
        panel = tk.Label(self.frame, text= str(file_name[len(file_name)-1]).upper()).pack()
        panel_image = tk.Label(self.frame, image=img)
        panel_image.image = img
        panel_image.pack()
        
    def get_caption(self):
        
        original = Image.open(self.image)
        original = original.resize((224, 224), Image.Resampling.LANCZOS)
        normal_image = img_to_array(original)
        # Pre-process the input images
        if self.feature_extraction_algo_name == 'VGG16':
            normal_image = tf.keras.applications.vgg16.preprocess_input(normal_image)
            with open(os.path.join(os.getcwd(), 'generated_data', 'metric_VGG16_LR_0.001_BATCH_SIZE_32_EPOCH_50.pkl'), 'rb') as file:
                data = load(file)
                tokenizer = data[11]
                max_caption_length = data[10]
        elif self.feature_extraction_algo_name == 'Xception':
            normal_image = tf.keras.applications.xception.preprocess_input(normal_image)
            with open(os.path.join(os.getcwd(), 'generated_data', 'metric_Xception_LR_0.001_BATCH_SIZE_32_EPOCH_50.pkl'), 'rb') as file:
                data = load(file)
                tokenizer = data[11]
                max_caption_length = data[10]
        image_id = self.image.split('/')[-1].split('.')[0]
        
        # feature = self.feature_extraction_model.predict(normal_image.reshape( (1,) + normal_image.shape[:3]))
        caption = self.ModelPrediction_Obj.predict_caption(self.features[image_id], tokenizer, max_caption_length)
        table = tk.Label(self.frame, text="Caption: " + caption[9:-7], font=("Helvetica", 12)).pack()
    
        return caption

if __name__=='__main__':

     # Create the parser for adding feature extraction algo name via command line agrument
    parser = argparse.ArgumentParser(description='Enter which feature extract algorithm to select for loading \
                                    right feature extraction model and subsequent trained caption generator \
                                    model')

    # Add an argument
    parser.add_argument('feature_extraction_algo_name', type=str, help=': Select from VGG16 or Xception')
    # Parse the arguments
    args = parser.parse_args()

    feature_extraction_algo_name = args.feature_extraction_algo_name
    caption_generator_model = None
    features = None

    # Load the feature extraction model
    if feature_extraction_algo_name == 'VGG16':
        with open(os.path.join(os.getcwd(), 'generated_data', 'features_VGG16.pkl'), 'rb') as file:
            features = load(file)
    elif feature_extraction_algo_name == 'Xception':
         with open(os.path.join(os.getcwd(), 'generated_data', 'features_Xception.pkl'), 'rb') as file:
            features = load(file)
    else:
        print('ERROR: No feature extraction model is loaded.  Choose the right one.')

   # Load the caption generator model
    if feature_extraction_algo_name == 'VGG16':
        print('file =', os.path.join(os.getcwd(), 'models', 'model.image_cation_generator_cnn_VGG16.keras'))
        model_file = os.path.join(os.getcwd(), 'models', 'model.image_cation_generator_cnn_VGG16.keras')
        caption_generator_model = load_model(model_file)
    elif feature_extraction_algo_name == 'Xception':
        model_file = os.path.join(os.getcwd(), 'models', 'model.image_cation_generator_cnn_Xception.keras')
        caption_generator_model = load_model(model_file)
    else:
        print('ERROR: No caption generator model is loaded.  Choose the right one.')

    root = tk.Tk()
    root.title('IMAGE CAPTION GENERATOR')
    # root.iconbitmap('cnn.png')
    root.resizable(False, False)
    tit = tk.Label(root, text="IMAGE CAPTION GENERATOR", padx=25, pady=6, font=("", 12)).pack()
    canvas = tk.Canvas(root, height=550, width=600, bg='#D1EDf2')
    canvas.pack()
    frame = tk.Frame(root, bg='white')
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    # Instatiate the class CaptionGeneratorGUI()
    CaptionGeneratorGUI_Obj = CaptionGeneratorGUI(caption_generator_model, 
                                                  features, 
                                                  feature_extraction_algo_name,
                                                  frame,
                                                  )

    chose_image = tk.Button(root, text='Choose Image',
                            padx=35, pady=10,
                            fg="black", bg="pink", command=CaptionGeneratorGUI_Obj.get_image, activebackground="#add8e6")
    chose_image.pack(side=tk.LEFT)

    caption_image = tk.Button(root, text='Classify Image',
                            padx=35, pady=10,
                            fg="black", bg="pink", command=CaptionGeneratorGUI_Obj.get_caption, activebackground="#add8e6")
    caption_image.pack(side=tk.RIGHT)
    root.mainloop()    