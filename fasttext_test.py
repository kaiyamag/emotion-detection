import io

# Number of word vectors to grab from pre-trained vectors. 
# Limited to 300,000 for debugging
NUM_WORDS = 300000

# TODO:
# Make function to get vector for a given word, else return ???
# Make function to make list of vectors from list of words/tokens

class InputProcessor:
    # Attributes:
    ft_filename = ""   # FastText word embeddings file
    # tokenized_str    # A list of Tokens
    vectorized_str = []  # A list of word vectors corresponding to words from tokenized_str
    vector_data = {}    # A dictionary where the key is a string and the value is a list of floats (300-D vector)

    # Initializer
    def __init__(self, ft_filename):
        self.ft_filename = ft_filename
    

    # From FastText documentation. Gets pre-trained word vectors. Populates and returns the vector_data
    # dictionary with words and their corresponding vector representations.
    def load_vectors(self):
        fin = io.open(self.ft_filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        self.vector_data = {}
        print(">> Finished opening file")

        i = 0
        for line in fin:
            if (i > NUM_WORDS):
                print(">> Exiting vector loading")
                break
            else:
                # Print status message every 100,000 words loaded
                if ((i % (NUM_WORDS / 5)) == 0):
                    print(">>", i, ", ", (i / NUM_WORDS) * 100, "% loaded")
                
                # Splits line on ' ' character, sets as list of tokens
                # vector format: one string followed by floats representing the vector, ("word" 123.45 543.2 456.678 ...)
                # Add vector of floats to dict[target word]. map() casts each item of remainder of tokens array to a float
                tokens = line.rstrip().split(' ')           
                self.vector_data[tokens[0]] = map(float, tokens[1:]) 
                i = i + 1
        return self.vector_data


    # Returns a 300-D vector (as a list) of floats corresponding to the given word. If the word
    # does not exist in the dictionary, returns a list of 0's
    def get_vector(self, word):
        if (word in self.vector_data):
            print(word, "is in the dictionary")
            return list(self.vector_data[word])     # Must convert map object to list before accessing
        else:
            print(word, "is not in the dictionary")
            empty = []
            for i in range(300):
                empty.append(0.0)
            return empty
    

    # Getter method. Returns dictionary of vector data
    def get_vector_data(self):
        return self.vector_data



def main():
    # Load word embeddings
    filename = "C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\wiki-news-300d-1M.vec"
    print("Loading word vectors from:", filename)
    processor = InputProcessor(filename)
    processor.load_vectors()
    vecs = processor.get_vector_data()
    print("Done loading vectors")
    print("--------------------------------------")

    # Test word vector retrieval 
    sample_vec = processor.get_vector("quite")
    print(sample_vec)
    test_word = "fdsa"
    print(test_word, len(processor.get_vector(test_word)))


if __name__ == '__main__':
    main()