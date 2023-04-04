import io
import numpy as np
import math
from InputProcessor import InputProcessor
from Token import Token
from GoEmotions import GoEmotions

# From ANN Course example
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
import keras

# From Confusion Matrix example (https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix)
from sklearn.metrics import confusion_matrix

# Number of comments to use from GoEmotions dataset
DATASET_SIZE = 1000

""" Manages the Tensorflow neural network model. 
"""
class Model:
    # Attributes:
    text = "A normal Denny's, Spires-like coffee shop in Los Angeles. It's about 9:00 in the morning. While the place isn't jammed, there's a healthy number of people drinking coffee, munching on bacon and eating eggs.Two of these people are a YOUNG MAN and a YOUNG WOMAN. The Young Man has a slight working-class English accent and, like his fellow countryman, smokes cigarettes like they're going out of style.It is impossible to tell where the Young Woman is from or how old she is; everything she does contradicts something she did. The boy and girl sit in a booth. Their dialogue is to be said in a rapid pace 'HIS GIRL FRIDAY' fashion."

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
    

    """Creates new Keras LSTM and compiles. Function from ANN Course example
    """
    def build_model(self):
        # Testing
        # chars = sorted(list(set(self.text)))
        # seqlen = 10
        comment_len = 30 # TODO: Remove magic numbers, replace w/ class constant or expression 
        word_embedding_len = 300
        emotion_vec_len = 28
        dropout_rate = 0.1

        # Create model
        self.model = Sequential()
        # model.add(LSTM(128, input_shape=(seqlen, len(chars)), return_sequences=True))     # Test w/ RNN and film script input

        # LSTM layer: Takes in vectors of shape (30, 300) and outputs vector of length 128
        self.model.add(LSTM(
            128, 
            input_shape=(comment_len, word_embedding_len), 
            activation='tanh',
            dropout=dropout_rate)
        )     

        # Output dense layer: Outputs a vector of length 28 with sigmoid activation 
        self.model.add(Dense(emotion_vec_len, activation='sigmoid'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=RMSprop(learning_rate=0.01),
            metrics=['categorical_crossentropy', 'accuracy']    # 'val_acc', 'val_loss'
        )

        return self.model
    
    
    """ Split x and y test sets from previously generated x and y train sets. Assumes x_train 
    and y_train are still in their original order. Takes test split as a percentage.
    """
    def build_test_sets(self, split):
        print("Building x and y test splits")

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
        print("Split index:", split_index)

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
        
        # DEBUG
        # print("First element of x_train:")
        # print(self.x_train[1])

        # TODO: Why is get_vectorized_str returning blank arrays?
        # Solution: replace [] with empty vec


    """ Trains model with x and y training data
    """
    def train_model(self, model):
        val_split = 0.1

        self.model.fit(self.x_train, self.y_train, 
            batch_size=128, 
            epochs=1,
            validation_split = val_split
        )

    
    """ Gets model prediction for given comment vector of shape (30, 300)
    """
    def get_pred(self, model, comment_vec):
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
    

    """ Converts a prediction vector of floats to a binary vector, for use in F1 score or confusion matrix
    """
    def to_binary(self, vec):
        binary_vec = list(map(lambda n: int(n >= 0.5), self.y_pred))
        print("Binary vec:", binary_vec)

        return binary_vec


    """ Prints a confusion matrix from expected output and actual output
    """
    def print_confusion_mat(self):
        print("Confusion matrix:")

        # Suggestion from https://stackoverflow.com/questions/48987959/classification-metrics-cant-handle-a-mix-of-continuous-multioutput-and-multi-la
        # argmax isn't relevant here: it gets the index of the maximum value in a numpy array
        # TODO: Better define what a "correct" output is
        adj_y_test = np.argmax(self.y_test, axis=1)
        adj_y_pred = np.argmax(self.y_pred, axis=1)

        # DEBUG
        print("adjusted y_test shape:", adj_y_test.shape)
        print("adjusted y_pred shape:", adj_y_pred.shape)
        # print("adjusted y_test:", adj_y_test)
        # print("adjusted y_pred:", adj_y_pred)

        # mat = confusion_matrix(self.y_test, self.y_pred)
        mat = confusion_matrix(adj_y_test, adj_y_pred)
        print(mat)

        return mat


""" Test model functions
"""
def main():
    my_model = Model()
    print("Done initializing model")

    my_model.build_train_sets()
    print("Done building x_train, shape", my_model.x_train.shape)
    print(my_model.x_train)

    print("\n\nDone building y_train, shape", my_model.y_train.shape)
    print(my_model.y_train)

    test_split = -50
    my_model.build_test_sets(test_split)
    print("Done building x_test, shape", my_model.x_test.shape)
    #print(my_model.x_test)

    print("\n\nDone building y_test, shape", my_model.y_test.shape)
    #print(my_model.y_test)

    print("Done reshaping x_train, shape", my_model.x_train.shape)
    print("Done reshaping y_train, shape", my_model.y_train.shape)

    my_model.build_model()
    print("\n\nDone building model")
    print(my_model.model.summary())

    my_model.train_model(my_model)
    print("Done training model")

    # # Make comment vector for prediction testing
    # str = "I am excited to eat pie"
    # tokenized_str = my_model.input_processor.tokenize(str)
    # comment_vec = my_model.input_processor.get_vectorized_str(tokenized_str)
    # comment_vec = np.array(comment_vec)[np.newaxis, :, :]
    # print("Comment: '", str, "' shape:", np.array(comment_vec).shape)

    # pred = my_model.get_pred(my_model, comment_vec)
    # print("Done getting prediction")

    # print("Prediction shape:", np.array(pred).shape)
    # print("Prediction:", pred)

    my_model.test_model()
    print("Done testing model, y_pred shape", my_model.y_pred.shape)
    print("y_pred array:", my_model.y_pred)
    print("One prediction:", my_model.y_pred[1])

    my_model.to_binary(my_model.y_pred)

    #my_model.print_confusion_mat()


if __name__ == '__main__':
    main()
