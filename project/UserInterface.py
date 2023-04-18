import io
import numpy as np
import math
from model import Model


""" Manages the user interface for interacting with the LSTM model. 
"""
class UserInterface:
    # Attributes:
    
    # Initializer
    def __init__(self):
        self.model = Model()
    
    def setup_model(self):
        self.model.build_train_sets()

        test_split = 0.2
        self.build_test_sets(test_split)
        self.model.build_model()
        self.model.train_model(self.model)
    

def main():
    print("Welcome to the LSTM model for detecting emotion in text!")
    print("Loading model...")

    ui = UserInterface()
    my_model = ui.model
    my_model.build_train_sets()
    test_split = Model.test_split
    my_model.build_test_sets(test_split)
    my_model.build_model()
    my_model.train_model(my_model)

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
