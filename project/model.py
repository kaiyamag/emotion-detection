import io
import numpy as np
from InputProcessor import InputProcessor
from Token import Token
from GoEmotions import GoEmotions

# From ANN Course example
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
import keras

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
        self.ge = GoEmotions()
        self.input_processor = InputProcessor(self.filename)
        self.model = Sequential()
    

    """Creates new Keras LSTM and compiles. Function from ANN Course example
    """
    def build_model(self):
        # Testing
        # chars = sorted(list(set(self.text)))
        # seqlen = 10
        comment_len = 30
        word_embedding_len = 300
        emotion_vec_len = 28

        # Create model
        self.model = Sequential()
        # model.add(LSTM(128, input_shape=(seqlen, len(chars)), return_sequences=True))     # Test w/ RNN and film script input
        self.model.add(LSTM(128, input_shape=(comment_len, word_embedding_len)))      # TODO: Remove 300 magic numbers, replace w/ class constant or expression 
        self.model.add(Dense(emotion_vec_len, activation='softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=RMSprop(learning_rate=0.01),
            metrics=['categorical_crossentropy', 'accuracy']
        )

        print(self.model.summary)

        self.model.fit(self.x_train, self.y_train, batch_size=128, epochs=1)

        return self.model
    
    
    """ Trains model with x and y training data
    """
    # def train_model(self, model):
        # model.fit(x_train, y_train, batch_size=128, epochs=1)


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

    my_model.build_model()
    print("Done building model")
    print(my_model.model.summary())

    # my_model.train_model(my_model)
    # print("Done training model")


if __name__ == '__main__':
    main()
