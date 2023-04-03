import tensorflow as tf
import tensorflow_datasets as tf_datasets
import numpy as np

from itertools import islice

from InputProcessor import InputProcessor

# Number of comments to use from GoEmotions dataset
DATASET_SIZE = 1000


""" This class stores the loaded GoEmotions database and manages functions to extract data from this dataset
"""
class GoEmotions:
    # Attributes
    dataset = []
    master_emotions = ['admiration',  'amusement',  'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',  'desire', 'disappointment', 'disapproval',  'disgust', 'embarrassment',  'excitement',  'fear', 'gratitude',  'grief', 'joy',  'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief', 'remorse',  'sadness',  'surprise']
    
    # For buildXTrain()
    x_train = []
    filename = "C:\\Users\\aeble\\Documents\\CS_200_Projects\\Junior_IS\\wiki-news-300d-1M.vec"
    input_processor = []


    # Initializer
    def __init__(self):
        self.dataset = tf_datasets.load('goemotions', split='train')
        self.input_processor = InputProcessor(self.filename)
        self.input_processor.load_vectors()
    

    """ Takes an index of this dataset. Returns the comment text of the datapoint at the given index. 
    """
    def extract_comment(self, index):
        # From https://stackoverflow.com/questions/16977385/extract-the-nth-key-in-a-python-dictionary/59740280#59740280
        # Gets dictionary item at index by iterating through dictionary
        iterator = iter(self.dataset)
        next(islice(iterator, index - 1, index - 1), None) 
        element = next(iterator)

        # Convert to utf-8 format
        comment_text = element['comment_text'].numpy()
        comment_text = comment_text.decode("utf-8")

        return comment_text

    
    """ Takes an index of this dataset. Returns a one-hot encoding of emotion labels for the datapoint at the given index. 
    """
    def extract_emotion_vec(self, index):
        emotions = []

        iterator = iter(self.dataset)
        next(islice(iterator, index - 1, index - 1), None) 
        element = next(iterator)

        for emotion in self.master_emotions:
            if (element[emotion].numpy() == True):
                emotions.append(1)
            else:
                emotions.append(0)

        return emotions
    

    """ Returns a list of all emotions present (value of 1) in the given emotion one-hot-encoding vector
    """
    def get_one_hot_emotions(self, vec):
        emotions = []

        # Check for invalid input
        if (not (len(vec) == len(self.master_emotions))):
            print("Length of given vector (", len(vec), "does not equal number of emotions", len(self.master_emotions))
            return None

        emotions = []
        for i in range(len(vec)):
            if (vec[i] == 1):
                emotions.append(self.master_emotions[i])
        
        return emotions


    """ Converts each extracted comment from the entire GoEmotions dataset to a list of vectorized comments
    """
    def buildXTrain(self):
        print("Building x_train dataset...")
        # For each entry in GoEmotions database, extract comment and convert to vector
        iterator = iter(self.dataset)
        if (DATASET_SIZE < len(self.dataset)):
            max = DATASET_SIZE
        else:
            max = len(self.dataset)

        for i in range(0, max):
            element = next(iterator)
            comment_text = element['comment_text'].numpy()
            comment_text = comment_text.decode("utf-8")
            comment_text = self.input_processor.tokenize(comment_text)
            
            comment_vec = self.input_processor.get_vectorized_str(comment_text)

            # DEBUG
            #print("comment:", element)
            #print("\nvec:", comment_vec)

            # Initialize x_train with first comment vec. Ensures all dataset elements have same numpy shape
            if (i == 0):
                self.x_train = np.array(comment_vec)[np.newaxis, :, :]
            else:
                #a = np.array(self.x_train)
                b = np.array(comment_vec)[np.newaxis, :, :]
                self.x_train = np.concatenate((self.x_train, b), axis=0)

                # DEBUG
                # print("------ i: ", i, "--------")
                # print("a shape: ", a.shape)
                # print("b shape: ", b.shape)

            # Print debugging message:
            if ((i % (max / 10)) == 0):
                print(">>", (i / max) * 100, "% complete ")
        
        # DEBUG
        # print("First element of x_train:")
        # print(self.x_train[1])

        # TODO: Why is get_vectorized_str returning blank arrays?
        # Solution: replace [] with empty vec


