import io

# Number of word vectors to grab from pre-trained vectors. 
# Limited to 300,000 for debugging
NUM_WORDS = 300000

def main():
    filename = "C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\wiki-news-300d-1M.vec"
    print("Loading word vectors from:", filename)
    # vector_data: a list of 300-D vectors of floats
    vector_data = load_vectors(filename)
    print("Done loading vectors")
    print("--------------------------------------")
    # Must convert map object to list before accessing
    print (len(list(vector_data["James"])));


# From FastText documentation. Gets pre-trained word vectors
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
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
            data[tokens[0]] = map(float, tokens[1:]) 
            i = i + 1
    return data

if __name__ == '__main__':
    main()