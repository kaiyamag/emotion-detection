import tensorflow as tf
import tensorflow_datasets as tf_datasets
from itertools import islice


""" This class stores the loaded GoEmotions database and manages functions to extract data from this dataset
"""
class GoEmotions:
    # Attributes
    dataset = []
    master_emotions = ['admiration',  'amusement',  'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',  'desire', 'disappointment', 'disapproval',  'disgust', 'embarrassment',  'excitement',  'fear', 'gratitude',  'grief', 'joy',  'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief', 'remorse',  'sadness',  'surprise']
    
    std_length = 28
    
    # Initializer
    def __init__(self):
        self.dataset = tf_datasets.load('goemotions', split='train')
    

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

    
    """ Takes an element of this dataset. Returns a one-hot encoding of emotion labels for the datapoint at the given index. 
    """
    def extract_emotion_from_element(self, element):
        emotions = []
        
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
            print("Length of given vector (", len(vec), ") does not equal number of emotions (", len(self.master_emotions), ")")
            return None

        emotions = []
        for i in range(len(vec)):
            if (vec[i] == 1):
                emotions.append(self.master_emotions[i])
        
        return emotions
