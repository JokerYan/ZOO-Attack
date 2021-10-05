from .post_model import PostModel


class PostModelTf():
    def __init__(self):
        self.post_model = PostModel()

    def predict(self, x):
        print(type(x))