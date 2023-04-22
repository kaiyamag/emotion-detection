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
    def __init__(self, filepath, model_file):
        self.model = Model(filepath, model_file)
    

    """ Prepares the model for predictions by loading training datasets, building, and training the model.
    """
    def setup_model(self):
        self.model.build_train_sets()

        test_split = 0.2
        self.model.build_test_sets(test_split)
        self.model.build_model()
        self.model.train_model()
    

""" Prompts the user for text to enter into the emotion detection model. Prints a list of emotions 
predicted for that text sample.
"""
def main():
    print("Welcome to the LSTM model for detecting emotion in text!")
    print("Loading model...")

    fastText_file = "C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\wiki-news-300d-1M.vec"
    model_file = "C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\emotion-detection\\model"

    ui = UserInterface(fastText_file, model_file)
    my_model = ui.model

    # Train model if no saved model was loaded
    if (not model_file):
        ui.setup_model()

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
