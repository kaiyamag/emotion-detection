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
#from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Adagrad
import keras

# From Confusion Matrix example (https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix)
# from sklearn.metrics import confusion_matrix

# Number of comments to use from GoEmotions dataset
DATASET_SIZE = 1000

""" Manages the Tensorflow neural network model. 
"""
class Model:
    # Attributes:

    # Model building
    comment_len = 30
    dropout_rate = 0.1
    lstm_size = 128
    lstm_actv = 'tanh'
    output_actv = 'softmax'
    learning_rate = 0.01

    # Model training
    validation_split = 0.1
    test_split = 0.2
    batch_size = 128
    num_epochs = 1

    # Binary threshold
    bin_threshold = 0.0357

    # Initializer
    def __init__(self):
        self.filename = "C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\wiki-news-300d-1M.vec"
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.y_pred = []
        self.ge = GoEmotions()
        self.input_processor = InputProcessor(self.filename)
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

        # Internal dense layers
        # self.model.add(Dense(512, activation='sigmoid'))

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
        # TEMPORARY
        emotion_counts = np.zeros(28, dtype=int)
        avg_num_emotions = 0
        sum_emotions = 0

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

            # DEBUG
            # print("comment:", element)
            # print("\nvec:", np.array(comment_vec))

            # Initialize x_train with first comment vec. Ensures all dataset elements have same numpy shape
            if (i == 0):
                self.x_train = np.array(comment_vec)[np.newaxis, :, :]
            
            # Add next vectorized comment to dataset
            else:
                #a = np.array(self.x_train)
                b = np.array(comment_vec)[np.newaxis, :, :]
                self.x_train = np.concatenate((self.x_train, b), axis=0)

                # DEBUG
                # print("------ i: ", i, "--------")
                # print("a shape: ", a.shape)
                # print("b shape: ", b.shape)
            
            # ---------- Y TRAIN ------------
            emotion_vec = self.ge.extract_emotion_from_element(element)

            if (i == 0):
                self.y_train = np.array(emotion_vec)[np.newaxis, :]
            else:
                b = np.array(emotion_vec)[np.newaxis, :]
                self.y_train = np.concatenate((self.y_train, b), axis=0)

                # DEBUG
                # print("------ i: ", i, "--------")
                # print("y_train shape: ", self.y_train.shape)
                # print("b shape: ", b.shape)

            # Print debugging message at every 10% interval:
            if ((i % (max / 10)) == 0):
                print(">>", (i / max) * 100, "% complete ")

            # TEMPORARY: Get histogram of emotion distribution in loaded dataset
            emotion_counts = np.array(emotion_counts + np.array(emotion_vec))
            sum_emotions = sum_emotions + np.sum(emotion_vec)
        
        # DEBUG
        # print("First element of x_train:")
        # print(self.x_train[1])

        # TODO: Why is get_vectorized_str returning blank arrays?
        # Solution: replace [] with empty vec

        print("Emotion counts:", emotion_counts)
        avg_num_emotions = sum_emotions / max
        print("Average num emotions:", avg_num_emotions)


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
    

    """ Converts a prediction vector of floats to a binary vector, for use in F1 score or confusion matrix.
    Takes a vector of floats as input and returns a vector of the same length of 1's and 0's.
    """
    @staticmethod
    def to_binary(vec):
        # THRESHOLD VALUE: any float greater than or equal to bin_threshold represents a positive identification of that emotion in the sample
        threshold = Model.bin_threshold
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

        # For prediction in y_pred
            # Look at corresponding expected y from y_test.
                # If expected[n] and prediction[n] are both 1, true_pos++
                # If only prediction[n] is 1, false_pos++
                # If expected[n] and prediction[n] are both 0, true_neg++
                # If only prediction[n] is 0, false_neg++
        
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


    """ Prints a confusion matrix from expected output and actual output
    """
    # def print_confusion_mat(self):
    #     print("Confusion matrix:")

    #     # Suggestion from https://stackoverflow.com/questions/48987959/classification-metrics-cant-handle-a-mix-of-continuous-multioutput-and-multi-la
    #     # argmax isn't relevant here: it gets the index of the maximum value in a numpy array
    #     # TODO: Better define what a "correct" output is
    #     #adj_y_test = np.argmax(self.y_test, axis=1)
    #     adj_y_test = self.y_test

    #     # Convert all predictions in y_pred to binary
    #     #adj_y_pred = np.argmax(self.y_pred, axis=1)
    #     adj_y_pred = []

    #     for pred in self.y_pred:
    #         adj_y_pred.append(self.to_binary(pred))
        
    #     adj_y_pred = np.array(adj_y_pred)

    #     # DEBUG
    #     print("adjusted y_test shape:", adj_y_test.shape)
    #     print("adjusted y_pred shape:", adj_y_pred.shape)
    #     print("adjusted y_test:", adj_y_test)
    #     print("adjusted y_pred:", adj_y_pred)

    #     # mat = confusion_matrix(self.y_test, self.y_pred)
    #     mat = confusion_matrix(adj_y_test, adj_y_pred)
    #     print(mat)

    #     return mat


""" Tests model functions.
"""
def main():
    my_model = Model()
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

    # my_model.build_model()
    # print("\n\nDone building model")
    # print(my_model.model.summary())

    # my_model.train_model(my_model)
    # print("Done training model")

    # # Make comment vector for prediction testing
    # str = "I am excited to eat pie"
    # tokenized_str = my_model.input_processor.tokenize(str)
    # comment_vec = my_model.input_processor.get_vectorized_str(tokenized_str)
    # comment_vec = np.array(comment_vec)[np.newaxis, :, :]
    # print("Comment: '", str, "' shape:", np.array(comment_vec).shape)

    # pred = my_model.get_pred(comment_vec)
    # print("Done getting prediction")

    # print("Prediction shape:", np.array(pred).shape)
    # print("Prediction:", pred)

    # my_model.test_model()
    # print("Done testing model, y_pred shape", my_model.y_pred.shape)
    # print("y_pred array:", my_model.y_pred)

    # binary_vec = Model.to_binary(my_model.y_pred[1])
    # print("Binary vec of size", len(binary_vec), ":", binary_vec)

    # # my_model.print_confusion_mat()

    # f1_score = Model.calculate_F1(my_model.y_pred, my_model.y_test)
    # print("F1 score:", f1_score)



""" Runs hardcoded parameter fine tuning tests on many variations of the model.
Prints F1-score and parameter configurations for each test in 'test_output.txt'
"""
def fine_tune(my_model):
    # Open output file
    file = open('test_output.txt', 'w')
    file.write("Test Output:\n")
    file.close()

    tests = {}

    # Generate test configurations
    # dropout_rate_default = 0.1
    # learning_rate_default = 0.01
    # batch_size_default = 128
    # num_epochs_default = 10
    # bin_threshold_default = 0.0357
    # lstm_size_default = 128

    # dropout_rate_set = [0.1, 0.3, 0.5]
    # learning_rate_set = [0.01, 0.05, 0.1]
    # batch_size_set = [64, 128, 512]
    # num_epochs_set = [20, 50, 100]
    # bin_threshold_set = [0.0357, 0.1, 0.5]   # 0.0357 = 1/28 
    # lstm_size_set = [128, 1024]

    dropout_rate_set = [0.3]
    learning_rate_set = [0.0005, 0.001]
    batch_size_set = [128]
    num_epochs_set = [10]
    bin_threshold_set = [0.1]   # 0.0357 = 1/28 
    lstm_size_set = [1024]

    configs = make_test(dropout_rate_set, learning_rate_set, batch_size_set, num_epochs_set, bin_threshold_set, lstm_size_set)

    # Shuffle configs to allow for selection of a random subset of configs to test
    random.shuffle(configs)
    print(len(configs), "Testing configurations:")

    i = 0
    lines = []

    # Test all possible binary threshold rates
    for config in configs[:1]:
        print(">>> Config", i, "of", len(configs), "<<<")
        i = i + 1

        # Update model configuration

        # comment_len
        # dropout_rate
        # lstm_size
        # lstm_actv
        # output_actv
        # learning_rate
        # validation_split
        # test_split
        # batch_size
        # num_epochs
        # bin_threshold 

        my_model.dropout_rate = config['dropout_rate']
        my_model.learning_rate = config['learning_rate']
        my_model.batch_size = config['batch_size']
        my_model.num_epochs = config['num_epochs']
        my_model.bin_threshold = config['bin_threshold']
        my_model.lstm_size = config['lstm_size']

        # params = {'dropout_rate': my_model.dropout_rate, 'learning_rate': my_model.learning_rate, 'batch_size': my_model.batch_size, 'num_epochs': my_model.num_epochs, 'bin_threshold': my_model.bin_threshold}
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
        if (i % 5 == 0):
            file = open('test_output.txt', 'a')
            file.writelines(lines)
            file.close()
            lines = []

            # COOLDOWN
            time.sleep(180)
        

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
