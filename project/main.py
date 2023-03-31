from InputProcessor import InputProcessor
from Token import Token
from GoEmotions import GoEmotions

import random

""" Currently demonstrates the process of loading GoEmotions and FastText models, extracting comment text and
emotions from test data, tokenizing a comment, and generating the word embedding vectors for a tokenized comment.
"""
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

    # First Draft Test
    a_text = "I love algorithms"
    a_emotion_vec = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
    a_vectorized = processor.get_vectorized_str(processor.tokenize(a_text))
    print("I love algorithms vectorized: \n", a_vectorized)



    # # Get a random test comment
    # index = (int) (random.random() * 1000)
    # sample_comment = ge_model.extract_comment(index)
    # print("Comment", index, ":", sample_comment)
    # sample_emotion_vec = ge_model.extract_emotion_vec(index)

    # # Get list of emotions associated with that comment
    # sample_emotions = ge_model.get_one_hot_emotions(sample_emotion_vec)
    # print("Sample emotions:", sample_emotions)

    # # Tokenize comment
    # sample_tokenized_str = processor.tokenize(sample_comment)
    # print("Tokenized string:", sample_tokenized_str)

    # # Get list of word embedding vectors associated with each token
    # vectorized_str = processor.get_vectorized_str(sample_tokenized_str)
    # print("Vectorized string:", vectorized_str)
    # print("length:", len(vectorized_str))


if __name__ == '__main__':
    main()