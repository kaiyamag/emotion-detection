import tensorflow as tf
import tensorflow_datasets as tf_datasets

def main():
    print("Hello world")
    dataset = tf_datasets.load('goemotions', split='train')
    
    # Directly from GoEmotions tutorial
    for element in dataset.take(1):
        print(element['comment_text'])
        print(element['joy'].dtype)     # TODO: Not actually outputting True/False :(

if __name__ == '__main__':
    main()

# TODO:
# Figure out how to re-format entries for reading (string + list of emotions)
# Figure out FastText's desired format
# Import FastText