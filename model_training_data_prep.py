from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class ModelTrainDataPrep():

    # class Init function
    def __init__(self):
        pass

    # Creating a list of image IDs
    def create_model_dataset(self, image_to_captions_mapping):

        # Creat a List of Image IDs
        image_ids = list(image_to_captions_mapping.keys())
        # Split into Training and Test Sets
        split = int(len(image_ids) * 0.70)
        train_data = image_ids[:split]
        remainder_split = int((len(image_ids) - split) * 0.66)
        validation_data = image_ids[split:split+remainder_split]
        test_data = image_ids[split+remainder_split:]

        return train_data, validation_data, test_data

    # Data generator function
    def data_generator(self, data_keys, image_to_captions_mapping, features, tokenizer, max_caption_length, vocab_size, batch_size):
        # Lists to store batch data
        X1_batch, X2_batch, y_batch = [], [], []
        # Counter for the current batch size
        batch_count = 0

        while True:
            # Loop through each image in the current batch
            for image_id in data_keys: 
                # Get the captions associated with the current image
                captions = image_to_captions_mapping[image_id]

                # Loop through each caption for the current image
                for caption in captions:
                    # Convert the caption to a sequence of token IDs
                    caption_seq = tokenizer.texts_to_sequences([caption])[0]

                    # Loop through the tokens in the caption sequence
                    for i in range(1, len(caption_seq)):
                        # Split the sequence into input and output pairs
                        in_seq, out_seq = caption_seq[:i], caption_seq[i]

                        # Pad the input sequence to the specified maximum caption length
                        in_seq = pad_sequences([in_seq], maxlen=max_caption_length)[0]

                        # Convert the output sequence to one-hot encoded format
                        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                        # Append data to batch lists
                        X1_batch.append(features[image_id][0])  # Image features
                        X2_batch.append(in_seq)  # Input sequence
                        y_batch.append(out_seq)  # Output sequence

                        # Increase the batch counter
                        batch_count += 1

                        # If the batch is complete, yield the batch and reset lists and counter
                        if batch_count == batch_size:
                            X1_batch, X2_batch, y_batch = np.array(X1_batch), np.array(X2_batch), np.array(y_batch)
                            yield [X1_batch, X2_batch], y_batch
                            X1_batch, X2_batch, y_batch = [], [], []
                            batch_count = 0    
