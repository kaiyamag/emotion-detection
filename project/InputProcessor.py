import io
import numpy as np
from Token import Token

# Number of word vectors to grab from pre-trained vectors. 
# Limited to 100,000 for debugging
NUM_WORDS = 100000

# Maximum length of comment text to process
MAX_STR_LENGTH = 30

""" Handles generation of list of word embeddings from FastText pre-trained embeddings. 
"""
class InputProcessor:
    # Attributes:
    ft_filename = ""   # FastText word embeddings file
    vectorized_str = []  # A list of word vectors corresponding to words from tokenized_str
    vector_data = {}    # A dictionary where the key is a string and the value is a list of floats (300-D vector)
    tokenized_str = []  # A list of tokens corresponding to comment text

    # Initializer
    def __init__(self, ft_filename):
        self.ft_filename = ft_filename
        self.load_vectors()
    

    """From FastText documentation. Gets pre-trained word vectors. Populates and returns the vector_data dictionary with 
    words and their corresponding vector representations.
    """
    def load_vectors(self):
        fin = io.open(self.ft_filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        self.vector_data = {}
        print(">> Finished opening file")

        i = 0
        for line in fin:
            if (i > NUM_WORDS - 1):
                print(">> Exiting vector loading")
                break
            else:
                # Print status message every 20% words loaded
                if ((i % (NUM_WORDS / 5)) == 0):
                    print(">>", i, ", ", (i / NUM_WORDS) * 100, "% loaded")
                
                # Splits line on ' ' character, sets as list of tokens
                # vector format: one string followed by floats representing the vector, ("word" 123.45 543.2 456.678 ...)
                # Add vector of floats to dict[target word]. map() casts each item of remainder of tokens array to a float
                tokens = line.rstrip().split(' ')           
                self.vector_data[tokens[0]] = map(float, tokens[1:]) 
                i = i + 1
        
        return self.vector_data


    """Returns a 300-D vector (as a list) of floats corresponding to the given word. If the word does not exist in the 
    dictionary, returns a list of 0's
    """
    def get_vector(self, token):
        word = token.get_word()

        # If word has a corresponding embedding vector, return it.
        if (word in self.vector_data):
            fetched_vec = list(self.vector_data[word])      # Must convert map object to list before accessing
            
            # DEBUG
            # print("Got word:", fetched_vec, "from token", word)

            # TODO: Ongoing bug: Even if word is in dictionary, it can fetch [] as the vector.
            # This workaround replaces all [] vectors with zeroes/empty vectors
            if (len(fetched_vec) != 0):
                return fetched_vec

        #print("Creating empty")
        # TODO: Should this return an empty list or a random vector? (check literature)
        empty = []
        for i in range(300):
            empty.append(0.0)

        return empty

        # TODO: Figure out why words not in first 10k are grabbed and == []
        # vector_data is an unsliceable dictionary


    """ Takes a list of up to MAX_STR_LENGTH Tokens. Populates and returns vectorized_str, a list of vector 
    representations of those tokens, filled with empty vectors if less than MAX_STR_LENGTH tokens were grabbed.
    """
    def get_vectorized_str(self, tokenized_str):
        strlen = 0
        self.vectorized_str = []

        for token in tokenized_str:
            if (strlen < MAX_STR_LENGTH):   # Add a word from the comment without exceeding the word limit
                vec = self.get_vector(token)
                self.vectorized_str.append(vec)
                strlen = strlen + 1

                # DEBUG
                #print("Token:", token, ", vector:", vec[:10])
            else:
                break

        # Make empty 300-D vector    
        empty = []
        for i in range(300):
            empty.append(0.0)
        
        # Fill remaining list spots with empty vectors, up to capacity of MAX_STR_LENGTH
        for i in range(MAX_STR_LENGTH - strlen):
            self.vectorized_str.append(empty)
        
        return self.vectorized_str
    

    """ Getter method. Returns dictionary of vector data
    """
    def get_vector_data(self):
        return self.vector_data

    
    """ Converts a string to a list of Tokens
    """
    def tokenize(self, str):
        str_list = str.split()
        self.tokenized_str = []

        for word in str_list:
            self.tokenized_str.append(Token(word))

        return self.tokenized_str


    """ Gets tokenized string
    """
    def get_tokenized_str(self):
        return self.tokenized_str
