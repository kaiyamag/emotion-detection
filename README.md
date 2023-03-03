# Jr. I.S. - Emotion Detection in Text with a Long Short Term Memory Model
Building off of my experience with neural networks, I plan to design a Long Short Term Memory (LSTM) model (a type of recurrent neural network) that takes a collection of word embeddings in a sentence as input and outputs a predicted emotion. I will use Google's GoEmotions database as my training data, FastText as my word embedding model, and the Keras library for building the LSTM model. Initially, my software will be a command-line interface where the user inputs a string of text and receives the top three emotion predictions and their percent confidence. However, one of my stretch goals is to develop a web GUI. Since Keras provides basic functions for training LSTMs, my software focus will be writing data preprocessing functions, experimenting with different model architectures, and writing a function to evaluate the accuracy of the model.

## Code Citations:

**Loading the GoEmotions dataset** Used example load function from GoEmotions documentation to implement GoEmotions class initializer.

**Loading the FastText dataset** Used example load function from FastText documentation to implement InputProcessor load_vectors function.

**Accessing the nth key of a dictionary** Took inspiration from a StackOverflow answer to get the nth key of a dictionary (https://stackoverflow.com/questions/16977385/extract-the-nth-key-in-a-python-dictionary/59740280#59740280).
