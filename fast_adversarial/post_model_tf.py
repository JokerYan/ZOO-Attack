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
        # print(x)

        x = tf.reshape(x, [-1, 32 * 32 * 3])

        y = tf.reduce_max(x, axis=1)
        print(y.shape)
        return y