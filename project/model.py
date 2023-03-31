import io
from InputProcessor import InputProcessor
from Token import Token
from GoEmotions import GoEmotions

# From ANN Course example
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
import keras


""" Manages the Tensorflow neural network model. 
"""
class Model:
    # Attributes:
    text = "A normal Denny's, Spires-like coffee shop in Los Angeles. It's about 9:00 in the morning. While the place isn't jammed, there's a healthy number of people drinking coffee, munching on bacon and eating eggs.Two of these people are a YOUNG MAN and a YOUNG WOMAN. The Young Man has a slight working-class English accent and, like his fellow countryman, smokes cigarettes like they're going out of style.It is impossible to tell where the Young Woman is from or how old she is; everything she does contradicts something she did. The boy and girl sit in a booth. Their dialogue is to be said in a rapid pace 'HIS GIRL FRIDAY' fashion."

    # Initializer
    def __init__(self):
        print("Init")
    
    # Creates new Keras LSTM and compiles. Function from ANN Course example
    def buildModel(self):
        # Testing
        chars = sorted(list(set(self.text)))
        seqlen = 10

        # Create model
        model = Sequential()
        model.add(LSTM(128, input_shape=(seqlen, len(chars)), return_sequences=True))
        model.add(Dense(len(chars), activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer=RMSprop(learning_rate=0.01),
            metrics=['categorical_crossentropy', 'accuracy']
        )


""" Test model functions
"""
def main():
    print("Hello world!")

    my_model = Model()
    print("Done initializing model")

    my_model.buildModel()
    print("Done building model")


if __name__ == '__main__':
    main()
