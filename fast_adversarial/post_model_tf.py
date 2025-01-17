import tensorflow as tf
import numpy as np
import torch

from tqdm import tqdm


from .post_model import PostModel


class PostModelTf():
    def __init__(self):
        self.post_model = PostModel()
        self.image_size = 32
        self.num_channels = 3
        self.num_labels = 10

    def predict(self, x):
        # print("===========================================")
        # print(type(x))
        # # print(x)
        #
        # x = tf.reshape(x, [-1, 32 * 32 * 3])
        # y = tf.reduce_max(x, axis=1)
        y = tf.py_function(func=self.py_predict, inp=[x], Tout=tf.float32)
        return y

    def py_predict(self, x):
        if not isinstance(x, np.ndarray):
            x = x.numpy()

        x = x.astype(np.float32)

        y_list = []
        batch_size = 64
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size, :, :, :]
            x_batch = torch.from_numpy(x_batch).cuda()
            # x_batch = x_batch.unsqueeze(0)
            # B x W x H x C -> B x C x W x H
            x_batch = x_batch.permute(0, 3, 1, 2)
            # if i == 0:
            #     self.post_model.update_post_model(x_batch)
            y_batch = self.post_model.forward(x_batch)
            y_list.append(y_batch.detach())
        y = torch.cat(y_list)
        # y = self.post_model.forward(x)
        y = y.detach().cpu().numpy()
        return y