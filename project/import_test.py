import tensorflow as tf
import tensorflow_datasets as tf_datasets


""" This class stores the loaded GoEmotions database and manages functions to extract data from this dataset
"""
class GoEmotions:
    # Attributes
    dataset = []
    master_emotions = ['admiration',  'amusement',  'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',  'desire', 'disappointment', 'disapproval',  'disgust', 'embarrassment',  'excitement',  'fear', 'gratitude',  'grief', 'joy',  'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief', 'remorse',  'sadness',  'surprise']

    # Initializer
    def __init__(self):
        self.dataset = tf_datasets.load('goemotions', split='train')
    

    """ Takes an index of this dataset. Returns the comment text of the datapoint at the given index. 
    """
    def extract_comment(self, index):
        container = self.dataset.take(index)

        for element in container:
            return element['comment_text'].numpy()

        return None     # Should not be reached
    

    
    """ Takes an index of this dataset. Returns a one-hot encoding of emotion labels for the datapoint at the given index. 
    """
    def extract_emotion_vec(self, index):
        container = self.dataset.take(index)
        emotions = []

        for element in container:
            print("Checking a new element....")
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


def main():
    # Directly from GoEmotions tutorial
    # `element` is a dictionary containing tensors for every emotion and the comment text
    # Tensor structure:
    # General:              tf.Tensor(`data`, shape=, dtype= )
    # GoEmotions Example:   tf.Tensor(False, shape=(), dtype=bool)
    # `.numpy() gets the data from the tensor
    print("---- First Test-----")
    dataset2 = tf_datasets.load('goemotions', split='train')

    # for element in dataset2.take(2):
    #     print(element['comment_text'].numpy())

    # element = list(dataset2)
    # print(element[0])

    element = next(iter(dataset2))
    print(element)
    print(element['comment_text'].numpy())

    print("------Second Test-------")
    ge_model = GoEmotions()

    comment_text = ge_model.extract_comment(1)
    print("Comment:", comment_text)

    emotion_vec = ge_model.extract_emotion_vec(1)
    print("Emotions:", emotion_vec)

    emotion_str = ge_model.get_one_hot_emotions(emotion_vec)
    print("Emotion strings:", emotion_str)

if __name__ == '__main__':
    main()
