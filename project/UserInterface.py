# UserInterface.py
# Kaiya Magnuson, April 2023

import io
import numpy as np
import math
from model import Model


""" Manages the user interface for interacting with the LSTM model. 
"""
class UserInterface:
    # Initializer
    def __init__(self, fastText, model_file, train_new):
        self.model_file = model_file

        if (train_new):
            self.model = Model(fastText, False)
            self.setup_model()
        else:
            self.model = Model(fastText, model_file)
    

    """ Prepares the model for predictions by loading training datasets, building, training, and saving the model.
    """
    def setup_model(self):
        self.model.build_train_sets()

        test_split = 0.2
        self.model.build_test_sets(test_split)
        self.model.build_model()
        self.model.train_model()
        self.model.save_model(self.model_file)
    

""" Prompts the user for text to enter into the emotion detection model. Prints a list of emotions 
predicted for that text sample.
"""
def main():
    print("Welcome to the LSTM model for detecting emotion in text!")
    print("Loading model...")

    #-------------- CHANGE WHEN RUNNING ON NEW COMPUTER -------------- 

    # fastText_file: the filepath of the FastText word embeddings dataset. Required if training a new model
    fastText_file = "C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\wiki-news-300d-1M.vec"

    # model_file: the filepath of a saved Keras model, or the desired file location for a new model to be saved
    model_file = "C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\emotion-detection\\model"

    # train_new: Set to true to train a new model and save at `model_file`, set to false to use model saved at `model_file`
    train_new = False

    # ----------------------------------------------------------------

    ui = UserInterface(fastText_file, model_file, train_new)
    my_model = ui.model

    prompt = "Please enter a sentence between 1 and 30 words or press 'q' to quit:\n"
    take_input = True

    while (take_input):
        text = input(prompt)

        if (text == "q"):
            print("Exiting program...")
            take_input = False
            break
        
        tokenized_str = my_model.input_processor.tokenize(text)
        vec = my_model.input_processor.get_vectorized_str(tokenized_str)
        vec = np.array(vec)[np.newaxis, :, :]
        pred = Model.to_binary(my_model.get_pred(vec)[0])
        emotions = my_model.ge.get_one_hot_emotions(pred)
        print("Predicted emotions:", emotions)


if __name__ == '__main__':
    main()
