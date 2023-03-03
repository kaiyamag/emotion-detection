from InputProcessor import InputProcessor
from Token import Token
from GoEmotions import GoEmotions

import random

def main():
    
    # Load word embeddings
    filename = "C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\wiki-news-300d-1M.vec"
    print("Loading word vectors from:", filename)
    processor = InputProcessor(filename)
    processor.load_vectors()
    vecs = processor.get_vector_data()
    print("Done loading vectors")
    print("--------------------------------------")

    print("Loading GoEmotions database...")
    ge_model = GoEmotions()
    print("Done loading GoEmotions")

    # Test conversion from list of words to list of vectors
    #sample_str = [Token("I"), Token("love"), Token("pie")] 

    # Get a random test comment
    index = (int) (random.random() * 1000)
    sample_comment = ge_model.extract_comment(index)
    print("Comment", index, ":", sample_comment)
    sample_emotion_vec = ge_model.extract_emotion_vec(index)
    sample_emotions = ge_model.get_one_hot_emotions(sample_emotion_vec)
    print("Sample emotions:", sample_emotions)

    # Tokenize comment
    sample_tokenized_str = processor.tokenize(sample_comment)
    print("Tokenized string:", sample_tokenized_str)

    vectorized_str = processor.get_vectorized_str(sample_tokenized_str)
    print("Vectorized string:", vectorized_str)
    print("length:", len(vectorized_str))


if __name__ == '__main__':
    main()