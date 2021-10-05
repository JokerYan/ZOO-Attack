import argparse
import os

import torch
import torch.nn as nn

from preact_resnet import PreActResNet18
from utils import get_loaders, get_train_loaders_by_class, post_train

pretrained_model_path = os.path.join('.', 'pretrained_models', 'cifar_model_weights_30_epochs.pth')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.set_defaults(mixup=True, type=bool)
    parser.add_argument('--no-mixup', dest='mixup', action='store_false')
    parser.add_argument('--pt-data', default='ori_neigh', choices=['ori_rand', 'ori_train', 'ori_neigh_train', 'ori_neigh', 'rand'], type=str)
    parser.add_argument('--pt-method', default='adv', choices=['adv', 'normal'], type=str)
    parser.add_argument('--pt-iter', default=5, type=int)
    parser.set_defaults(rs_neigh=True, type=bool)
    parser.add_argument('--no-rs-neigh', dest='rs_neigh', action='store_false')
    parser.set_defaults(blackbox=False, type=bool)
    parser.add_argument('--blackbox', dest='blackbox', action='store_true')
    return parser.parse_args()


class PostModel(nn.Module):
    def __init__(self, model=None, args=None):
        super().__init__()

        if model is None:
            state_dict = torch.load(pretrained_model_path)
            model = PreActResNet18().cuda()
            model.load_state_dict(state_dict)
            model.float()
            model.eval()
        self.model = model

        if args is None:
            args = get_args()
        self.args = args

        self.train_loader, _ = get_loaders(self.args.data_dir, batch_size=128)
        self.train_loaders_by_class = get_train_loaders_by_class(self.args.data_dir, batch_size=128)

    def forward(self, images):
        post_model, original_class, neighbour_class, loss_list, acc_list, neighbour_delta = \
            post_train(self.model, images, self.train_loader, self.train_loaders_by_class, self.args)
        return post_model(images)
