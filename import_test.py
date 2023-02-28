import tensorflow as tf
import tensorflow_datasets as tf_datasets

def main():
    print("Hello world")
    dataset = tf_datasets.load('goemotions', split='train')
    
    # Directly from GoEmotions tutorial
    # `element` is a dictionary containing tensors for every emotion and the comment text
    for element in dataset.take(1):     
        print("\nFull element:\n")
        print(element)

        print("\nExtracted comment text:\n")
        print(element['comment_text'])

        # Tensor structure:
        # General:              tf.Tensor(`data`, shape=, dtype= )
        # GoEmotions Example:   tf.Tensor(False, shape=(), dtype=bool)
        print("\nExtracted joy T/F:\n")
        tensor = element['joy']
        print(tensor.numpy())     # `.numpy() gets the data from the tensor`

if __name__ == '__main__':
    main()

# TODO:
# Figure out how to re-format entries for reading (string + list of emotions)
# Figure out FastText's desired format
# Import FastText