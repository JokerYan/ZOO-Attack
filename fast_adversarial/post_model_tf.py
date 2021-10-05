import tensorflow as tf


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
        batch_size = tf.size(x)[0]
        print(tf.size(x))
        print(batch_size)

        return tf.random.uniform(shape=[batch_size, self.num_labels])