from InputProcessor import InputProcessor

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
    # sample_vec = processor.get_vector("quite")
    # print(sample_vec)
    # test_word = "fdsa"
    # print(test_word, len(processor.get_vector(test_word)))

    # Test conversion from list of words to list of vectors
    sample_str = ["quite", "woman", "James"] 
    vectorized_str = processor.get_vectorized_str(sample_str)
    print("Vectorized string:", vectorized_str)
    print("length:", len(vectorized_str))


if __name__ == '__main__':
    main()