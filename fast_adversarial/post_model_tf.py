import tensorflow as tf
import numpy as np


from .post_model import PostModel


class PostModelTf():
    def __init__(self):
        self.post_model = PostModel()
        self.image_size = 32
        self.num_channels = 3
        self.num_labels = 10

    def predict(self, x):
        print("===========================================")
        print(type(x))
        shape = x.get_shape().as_list()
        batch_size = shape[0]
        print(int(batch_size))

        return tf.convert_to_tensor(np.zeros([batch_size, self.num_labels]))