# Junior Independent Study - Emotion Detection in Text with a Long Short Term Memory Model
### Kaiya Magnuson, April 2023
The goal of this project is to build a machine learning model for emotion detection in short text samples. Through a command-line interface, users enter a sentence of up to 30 words, and the model prints a list of all the emotions identified in the text. A Long Short-Term Memory model is used due to its strength in processing sequence data, such as sentences. The machine learning model is trained on the Google GoEmotions dataset of Reddit comments labelled with emotions and converted to word vectors using transfer learning from the FastText word embedding model. Since Keras provides basic functions for training LSTMs, my software focus was writing data preprocessing functions, experimenting with different model architectures, and writing a function to evaluate the accuracy of the model.

## Running the Emotion Detection Model
The model is initialized and accessed through the `UserInterface` class. Install all dependencies listed in `requirements.txt`. Download the FastText word embedding dataset from [Mikolov et al.](https://fasttext.cc/index.html). Update the filepath of the FastText word embeddings dataset the model save location in the `main()` function of `UserInterface.py` and in `unit_tests.py`. Run `UserInterface.py` and follow prompts in the command-line interface to enter text and recieve emotion predictions. To run unit tests, run `unit_tests.py`, which will print out functions that fail the tests. Model fine-tuning tests can be run by running the main function in `model.py`.

## Class Documentation
This section explains the role of each class and its functions.

### UserInterface
Manages the user interface for interacting with the LSTM model.

*No class attributes*

**model:** An instance of the `Model` class.

**model_file:** The filepath of the saved Keras model. If given at initialization, that model will be used for the user interface. Otherwise, a new model will be trained according to the hyperparameters in `model.py`.

*Class functions:* 

**init(fastText, model_file, train_new):** Initializes a new `Model` instance, to be used as the primary model for all training and predictions. Takes the filepath of the FastText word embedding dataset and False for `model_file` if a new model is being trained, or False for `fastText` and the filepath of the saved Keras model if the model is being loaded from a pre-existing model. If `train_new` is true, trains a new model using `fastText_file` and saves it at `model_file`. If `train_new` is false, loads a pretrained model from `model_file`.

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

**init(self, fastText, model_file):** Defines and intializes class attributes. Takes the filepath of the FastText word embedding dataset and False for `model_file` if a new model is being trained, or False for `fastText` and the filepath of the saved Keras model if the model is being loaded from a pre-existing model.

**build_model(self):** Creates a new Keras LSTM and compiles it.

**build_test_sets(self, split):** Splits x and y test sets from previously generated x and y train sets. Assumes x_train and y_train are still in their original order. Takes test split as a percentage.

**build_train_sets(self):** Creates x training dataset of vectorized comments. 

**train_model(self):** Trains model with x and y training data.

**save_model(self, filepath):** Saves a copy of the trained model at the given filepath.

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

**init(self, ft_filename, reload_data):** Creates a new `InputProcessor` instance from given FastText word embeddings file location. Calls load_vectors. If `reload_data` is true, loads FastText word embeddings for model training.

**load_vectors(self):** From FastText documentation. Gets pre-trained word vectors. Populates and returns the vector_data dictionary with words and their corresponding vector representations.

**get_vector(self, token):** Returns a 300-D vector (as a list) of floats corresponding to the given word. If the word does not exist in the dictionary, returns a list of 0's.

**get_vectorized_str(self, tokenized_str):** Takes a list of up to `MAX_STR_LENGTH` `Tokens`. Populates and returns `vectorized_str`, a list of vector representations of those tokens, filled with empty vectors if less than `MAX_STR_LENGTH` tokens were grabbed.

**get_vector_data(self):** Getter method. Returns dictionary of vector data.

**tokenize(self, str):** Converts a string to a list of `Tokens`. Returns a list of `Tokens`.

**get_tokenized_str(self):** Gets tokenized string.


### GoEmotions
Stores the loaded GoEmotions database and manages functions to extract data from this dataset.

*Class attributes*

**dataset:** The set of text samples and corresponding emotion labels from the GoEmotions dataset.

*Class functions:* 

**init(self, reload_data):** Creates a new GoEmotions instance. If `reload_data` is true, loads GoEmotions data for model training.
 
**extract_comment(self, index):** Takes an index of this dataset. Returns the comment text of the datapoint at the given index. 

**extract_emotion_vec(self, index):** Takes an index of this dataset. Returns a one-hot encoding of emotion labels for the datapoint at the given index. 

**extract_emotion_from_element(self, element):** Takes an element of this dataset. Returns a one-hot encoding of emotion labels for the datapoint at the given index. 

**get_one_hot_emotions(self, vec):** Returns a list of all emotions present (value of 1) in the given emotion one-hot-encoding vector.


### Token
Represents a single word token.

*Class attributes*

**word:** A string representation of the word stored in this Token

*Class functions:* 

**init(self, word):** Creates a new Token with the given word.
 
**get_word(self):** Returns the string stored in this token.

**repr(self):** Prints a Token as its string representation.


### unit_tests.py
A suite of functions for testing dataset loading. See `unit_tests.py` for specific function documentation

*File functions:* 

**main():** Runs all unit tests in `unit_tests.py`, testing `InputProcessor`, `GoEmotions`, `Token` and some `Model` functions. Prints a summary of which tests were passed. 
 

## Code Citations:

**Loading the GoEmotions dataset:** Used example load function from [GoEmotions documentation](https://github.com/tensorflow/models/blob/fa3ba13e2b16782f3b0f483d24f4110877264e61/research/seq_flow_lite/demo/colab/emotion_colab.ipynb) to implement GoEmotions class initializer.

**Loading the FastText dataset:** Used example load function from [FastText documentation](https://fasttext.cc/docs/en/english-vectors.html) to implement InputProcessor load_vectors function.

**Accessing the nth key of a dictionary:** Took inspiration from [a StackOverflow answer](https://stackoverflow.com/questions/16977385/extract-the-nth-key-in-a-python-dictionary/59740280#59740280) to get the nth key of a dictionary .

**Artificial Neural Networks course example:** Followed the build_model() function from a course template. Template by Lucian Leahu, DIS Study Abroad.
