# Jr. I.S. - Emotion Detection in Text with a Long Short Term Memory Model
The goal of this project is to build a machine learning model for emotion detection in short text samples. Through a command-line interface, users enter a sentence of up to 30 words, and the model prints a list of all the emotions identified in the text. A Long Short-Term Memory model is used due to its strength in processing sequence data, such as sentences. The machine learning model is trained on the Google GoEmotions dataset of Reddit comments labelled with emotions and converted to word vectors using transfer learning from the FastText word embedding model. Since Keras provides basic functions for training LSTMs, my software focus was writing data preprocessing functions, experimenting with different model architectures, and writing a function to evaluate the accuracy of the model.

## Running the Emotion Detection Model
The model is initialized and accessed through the `UserInterface` class. Install all dependencies listed in `requirements.txt`. Run `UserInterface.py` and follow prompts in the command-line interface to enter text and recieve emotion predictions. To run unit tests, run `unit_tests.py`, which will print out functions that fail the tests. Model fine-tuning tests can be run by running the main function in `model.py`.

## Class Documentation
This section explains the role of each class and its functions.

### UserInterface
Manages the user interface for interacting with the LSTM model.

*No class attributes*

*Class functions:* 

**init():** Initializes a new `Model` instance, to be used as the primary model for all training and predictions.

**setup_model(self):** Prepares the model for predictions by loading training datasets, building, and training the model.

**main():** Prompts the user for text to enter into the emotion detection model. Prints a list of emotions predicted for that text sample.

### Model
Manages the user interface for interacting with the LSTM model.

*Class attributes:*

**filename:** File location for FastText word embedding vectors.

**x_train:** Stores the x training data (word embedding representation of text samples) 

**y_train:** Stores the y training data (emotion vectors corresponding to text samples)

**x_test:** Stores the x testing data (word embedding representation of text samples) 

**y_test:** Stores the y testing data (emotion vectors corresponding to text samples)

**y_pred** Stores y predictions (emotion vectors of floats corresponding to text samples)

**ge:** An instance of the `GoEmotions` class for data processing

**input_processor:** An instance of the `InputProcessor` class for data processing

**model:** Stores a Keras Sequential model


*Class functions:*

**init():** Defines and intializes class attributes.

**build_model(self):** Creates a new Keras LSTM and compiles it.

**build_test_sets(self, split):** Splits x and y test sets from previously generated x and y train sets. Assumes x_train and y_train are still in their original order. Takes test split as a percentage.

**build_train_sets(self):** Creates x training dataset of vectorized comments. 

**train_model(self):** Trains model with x and y training data.

**get_pred(self, comment_vec):** Gets model prediction for given comment vector of shape (30, 300).

**test_model(self):** Runs model with testing datasets. Returns array of output emotion vector preditions.

**to_binary(vec):** Static method. Converts a prediction vector of floats to a binary vector, for use in F1 score or confusion matrix. Takes a vector of floats as input and returns a vector of the same length of 1's and 0's.

**calculate_F1(y_pred, y_test):** Static method. Calculates a F1 score for the predicted emotions. Takes a list of emotion predictions as vectors and a list of expected emotion vectors. Returns the F1 score as a float.

**main():** Tests model functions.

**fine_tune(my_model):** Runs hardcoded parameter fine tuning tests on many variations of the model. Takes an instance of a `Model`. Prints F1-score and parameter configurations for each test in `test_output.txt`.

**make_test(dropout_rate_set, learning_rate_set, batch_size_set, num_epochs_set, bin_threshold_set):** Returns a list of all possible parameter configurations using given parameter sets.


### InputProcessor
Handles generation of list of word embeddings from FastText pre-trained embeddings. 

*Global class constants:*

**NUM_WORDS:** Number of word vectors to grab from pre-trained vectors. Limited to 100,000 for debugging.

**MAX_STR_LENGTH:** Maximum length of comment text to process.

*Class attributes:*

**ft_filename:** FastText word embeddings file location.

**tokenized_str:** A list of tokens corresponding to comment text.

**vectorized_str:** A list of word vectors corresponding to words from tokenized_str.

**vector_data:** A dictionary where the key is a string and the value is a list of floats (300-D vector).

*Class functions:*

**init(self, ft_filename):** Creates a new `InputProcessor` instance from given FastText word embeddings file location. Calls load_vectors.

**load_vectors(self):** From FastText documentation. Gets pre-trained word vectors. Populates and returns the vector_data dictionary with words and their corresponding vector representations.

**get_vector(self, token):** Returns a 300-D vector (as a list) of floats corresponding to the given word. If the word does not exist in the dictionary, returns a list of 0's.

**get_vectorized_str(self, tokenized_str):** Takes a list of up to `MAX_STR_LENGTH` `Tokens`. Populates and returns `vectorized_str`, a list of vector representations of those tokens, filled with empty vectors if less than `MAX_STR_LENGTH` tokens were grabbed.

**get_vector_data(self):** Getter method. Returns dictionary of vector data.

**tokenize(self, str):** Converts a string to a list of `Tokens`. Returns a list of `Tokens`.

**get_tokenized_str(self):** Gets tokenized string.


## Code Citations:

**Loading the GoEmotions dataset:** Used example load function from [GoEmotions documentation](https://github.com/tensorflow/models/blob/fa3ba13e2b16782f3b0f483d24f4110877264e61/research/seq_flow_lite/demo/colab/emotion_colab.ipynb) to implement GoEmotions class initializer.

**Loading the FastText dataset:** Used example load function from [FastText documentation](https://fasttext.cc/docs/en/english-vectors.html) to implement InputProcessor load_vectors function.

**Accessing the nth key of a dictionary:** Took inspiration from [a StackOverflow answer](https://stackoverflow.com/questions/16977385/extract-the-nth-key-in-a-python-dictionary/59740280#59740280) to get the nth key of a dictionary .

**Artificial Neural Networks course example:** Followed the build_model() function from a course template. Template by Lucian Leahu, DIS Study Abroad.
