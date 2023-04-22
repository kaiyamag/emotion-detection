# model.py
# Kaiya Magnuson, April 2023

import io
import numpy as np
import math
import random
import time
from InputProcessor import InputProcessor
from Token import Token
from GoEmotions import GoEmotions

# From ANN Course example
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
import keras

# Number of comments to use from GoEmotions dataset
DATASET_SIZE = 1000

""" Manages the Tensorflow neural network model. 
"""
class Model:
    # Attributes:

    # Model building
    comment_len = 15
    dropout_rate = 0.1
    lstm_size = 128
    lstm_actv = 'tanh'
    output_actv = 'softmax'
    learning_rate = 0.0005

    # Model training
    validation_split = 0.1
    test_split = 0.2
    batch_size = 128
    num_epochs = 1

    # Binary threshold
    bin_threshold = 0.1

    # Initializer
    def __init__(self, fastText, model_file):
        if (model_file):
            reload_data = False
        else:
            reload_data = True

        self.filename = fastText
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.y_pred = []
        self.ge = GoEmotions(reload_data)
        self.input_processor = InputProcessor(self.filename, reload_data)

        # Load saved model if applicable
        if (model_file):
            self.model = keras.models.load_model(model_file)
        else:
            self.model = Sequential()
    

    """ Creates a new Keras LSTM and compiles it. Function from ANN Course example
    """
    def build_model(self):
        comment_len = self.comment_len
        word_embedding_len = self.input_processor.std_length
        emotion_vec_len = self.ge.std_length
        dropout_rate = self.dropout_rate

        # Create model
        self.model = Sequential()
        
        # LSTM layer: Takes in vectors of shape (30, 300) and outputs vector of length lstm_size
        self.model.add(LSTM(
            self.lstm_size, 
            input_shape=(comment_len, word_embedding_len), 
            activation=self.lstm_actv,
            dropout=dropout_rate)
        )     

        # Output dense layer: Outputs a vector of length 28 with softmax activation 
        self.model.add(Dense(emotion_vec_len, activation=self.output_actv))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['categorical_crossentropy', 'accuracy']
        )

        return self.model
    
    
    """ Split x and y test sets from previously generated x and y train sets. Assumes x_train 
    and y_train are still in their original order. Takes test split as a percentage.
    """
    def build_test_sets(self, split):
        # Check that given split is a percentage between 0 and 1. Split defaults to 0.2 if exception is thrown
        try:
            assert (split >= 0 and split <= 1.0), "Testing split must be between 0 and 1."
        except AssertionError as exc:
            print(">> Error:", exc)
            split = 0.2

        # Check that x and y train sets are the same length. Exits function if exception is thrown
        try:
            assert len(self.x_train) == len(self.y_train), "x_train and y_train have different lengths."
        except AssertionError as exc:
            print(">> Error:", exc)
            return -1

        split_index = math.floor(split * len(self.x_train))

        # Gets split% last elements of x and y train sets to use for test sets
        self.x_test = self.x_train[-split_index:]
        self.y_test = self.y_train[-split_index:]

        # Truncate x/y train sets to [:split index] so that train and test sets don't overlap
        self.x_train = self.x_train[:-(split_index + 1)]
        self.y_train = self.y_train[:-(split_index + 1)]


    """ Create x training dataset of vectorized comments. 
    """
    def build_train_sets(self):
        print("Building x and y training datasets...")

        # Set max dataset size
        iterator = iter(self.ge.dataset)
        if (DATASET_SIZE < len(self.ge.dataset)):
            max = DATASET_SIZE
        else:
            max = len(self.ge.dataset)

        # For each entry in GoEmotions database, extract comment and convert to vector
        for i in range(0, max):
            # ------------- X TRAIN -------------
            element = next(iterator)
            comment_text = element['comment_text'].numpy()
            comment_text = comment_text.decode("utf-8")
            comment_text = self.input_processor.tokenize(comment_text)
            comment_vec = self.input_processor.get_vectorized_str(comment_text)

            # Initialize x_train with first comment vec. Ensures all dataset elements have same numpy shape
            if (i == 0):
                self.x_train = np.array(comment_vec)[np.newaxis, :, :]
            
            # Add next vectorized comment to dataset
            else:
                b = np.array(comment_vec)[np.newaxis, :, :]
                self.x_train = np.concatenate((self.x_train, b), axis=0)
            
            # ---------- Y TRAIN ------------
            emotion_vec = self.ge.extract_emotion_from_element(element)

            if (i == 0):
                self.y_train = np.array(emotion_vec)[np.newaxis, :]
            else:
                b = np.array(emotion_vec)[np.newaxis, :]
                self.y_train = np.concatenate((self.y_train, b), axis=0)

            # Print debugging message at every 10% interval:
            if ((i % (max / 10)) == 0):
                print(">>", (i / max) * 100, "% complete ")


    """ Trains model with x and y training data
    """
    def train_model(self):
        self.model.fit(self.x_train, self.y_train, 
            batch_size=self.batch_size, 
            epochs=self.num_epochs,
            validation_split = self.validation_split
        )

    
    """ Gets model prediction for given comment vector of shape (30, 300)
    """
    def get_pred(self, comment_vec):
        pred = self.model.predict(comment_vec, verbose=0)

        return pred


    """ Runs model with testing datasets. Returns array of output emotion vector preditions.
    """
    def test_model(self):
        print("Testing model with x_test set")

        self.y_pred = []

        # Check that x and y test sets are the same length. Exits function if exception is thrown
        try:
            assert len(self.x_test) == len(self.y_test), "x_test and y_test have different lengths."
        except AssertionError as exc:
            print(">> Error:", exc)
            return -1
        
        self.y_pred = (self.model.predict(self.x_test))
        
        return self.y_pred
    

    """ Saves trained model at specified file location.
    """
    def save_model(self, filepath):
        self.model.save(filepath)
        print("Model saved")
    

    """ Converts a prediction vector of floats to a binary vector, for use in F1 score or confusion matrix.
    Takes a vector of floats as input and returns a vector of the same length of 1's and 0's.
    """
    @staticmethod
    def to_binary(vec):
        # THRESHOLD VALUE: any float greater than or equal to bin_threshold represents a positive identification of that emotion in the sample
        # threshold = Model.bin_threshold
        # binary_vec = list(map(lambda n: int(n >= threshold), vec))

        # METHOD 2: Get single max value:
        threshold = np.amax(np.array(vec))
        binary_vec = list(map(lambda n: int(n >= threshold), vec))
       
        return binary_vec

    """ Calculates a F1 score for the predicted emotions. Takes a list of emotion predictions as vectors
    and a list of expected emotion vectors. Returns the F1 score as a float.
    """
    @staticmethod
    def calculate_F1(y_pred, y_test):
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        # Precision = # True Positives / (# True Positives + # False Positives)
        # Recall = # True Positives / (# True Positives + # False Negatives)
        # True Positives = # of comments that had a predicted emotion of x, and the test data also had that emotion
        # False Positives = # of comments that had a predicted emotion of x, but the test data did not have that emotion
        # False Negatives = # of comments that did not have a predicted emotion of x, but the test data did have that emotion

        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        
        try:
            assert (len(y_pred) == len(y_test)), "y_pred and y_test must be the same length"
        except AssertionError as exc:
            print("Error:", exc)
            return -1
        
        for i in range(len(y_pred)):
            expected = y_test[i]
            prediction = np.array(Model.to_binary(y_pred[i]))

            try:
                assert (len(expected) == len(prediction)), "Prediction and test vector must be the same length"
            except AssertionError as exc:
                print("Error:", exc)
                return -1

            for n in range(len(expected)):
                if (expected[n] == 1 and prediction[n] == 1):   # True positive detected
                    true_pos = true_pos + 1
                elif (prediction[n] == 1):  # False positive detected
                    false_pos = false_pos + 1
                
                if (expected[n] == 0 and prediction[n] == 0):   # True negative detected
                    true_neg = true_neg + 1
                elif (prediction[n] == 0):  # False negative detected
                    false_neg = false_neg + 1
        
        # Calculate F1 Score and check for divide by zero errors
        if (true_pos + false_pos == 0):
            print("Divide by zero error while calculating F1-score")
            return 0
        
        if (true_pos + false_neg == 0):
            print("Divide by zero error while calculating F1-score")
            return 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        if (precision + recall == 0):
            print("Divide by zero error while calculating F1-score")
            return 0

        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return f1_score


""" Tests model functions.
"""
def main():
    fastText = "C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\wiki-news-300d-1M.vec"
    my_model = Model(fastText, False)
    print("Done initializing model")

    my_model.build_train_sets()
    print("Done building x_train, shape", my_model.x_train.shape)
    print(my_model.x_train)

    print("\n\nDone building y_train, shape", my_model.y_train.shape)
    print(my_model.y_train)

    test_split = Model.test_split
    my_model.build_test_sets(test_split)
    print("Done building x_test, shape", my_model.x_test.shape)

    print("\n\nDone building y_test, shape", my_model.y_test.shape)

    print("Done reshaping x_train, shape", my_model.x_train.shape)
    print("Done reshaping y_train, shape", my_model.y_train.shape)

    #---------------- Transition to Fine Tuning --------------------

    fine_tune(my_model)

    my_model.save_model("C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\emotion-detection\\model")


""" Runs hardcoded parameter fine tuning tests on many variations of the model.
Prints F1-score and parameter configurations for each test in 'test_output.txt'
"""
def fine_tune(my_model):
    # Open output file
    file = open('test_output.txt', 'w')
    file.write("Test Output:\n")
    file.close()

    tests = {}

    dropout_rate_set = [0.1]
    learning_rate_set = [ 0.0005]
    batch_size_set = [128]
    num_epochs_set = [50]
    bin_threshold_set = [0.1]   # 0.0357 = 1/28 
    lstm_size_set = [128]

    configs = make_test(dropout_rate_set, learning_rate_set, batch_size_set, num_epochs_set, bin_threshold_set, lstm_size_set)

    # Shuffle configs to allow for selection of a random subset of configs to test
    random.shuffle(configs)
    print(len(configs), "Testing configurations:")

    i = 0
    lines = []

    # Test all possible binary threshold rates
    for config in configs[:300]:
        print(">>> Config", i, "of", len(configs), "<<<")
        i = i + 1

        # Update model configuration
        my_model.dropout_rate = config['dropout_rate']
        my_model.learning_rate = config['learning_rate']
        my_model.batch_size = config['batch_size']
        my_model.num_epochs = config['num_epochs']
        my_model.bin_threshold = config['bin_threshold']
        my_model.lstm_size = config['lstm_size']

        params = config

        # Build, train, and test model
        my_model.build_model()
        my_model.train_model()
        my_model.test_model()

        # Add model score to test results
        f1_score = Model.calculate_F1(my_model.y_pred, my_model.y_test)
        tests[f1_score] = params

        temp = str(f1_score) + str(params) + "\n"
        lines.append(temp)

        # Write line buffer to file
        buffer_len = 1
        cooldown = 0
        if (i % buffer_len == 0):
            file = open('test_output.txt', 'a')
            file.writelines(lines)
            file.close()
            lines = []

            # COOLDOWN
            time.sleep(cooldown)
        

    print("Fine-tuning test results:")

    # Print a dictionary code from Geeks for Geeks
    for i in sorted(tests.keys(), reverse=True):
        print(i, tests[i])
    
    print("Remaining configs:")
    print(configs[300:])
    

""" Returns a list of all possible parameter configurations using given parameter sets.
"""
def make_test(dropout_rate_set, learning_rate_set, batch_size_set, num_epochs_set, bin_threshold_set, lstm_size_set):
    configs = []

    # Lists every combination of parameters in given sets. 
    for dr in dropout_rate_set:
        for lr in learning_rate_set:
            for b in batch_size_set:
                for e in num_epochs_set:
                    for bt in bin_threshold_set:
                        for ls in lstm_size_set:
                            temp = {'dropout_rate': dr, 'learning_rate': lr, 'batch_size': b, 'num_epochs': e, 'bin_threshold': bt, 'lstm_size': ls}
                            configs.append(temp)

    return configs

if __name__ == '__main__':
    main()
