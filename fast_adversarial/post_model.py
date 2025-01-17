import argparse
import os

import torch
import torch.nn as nn
from torchvision import transforms

from .preact_resnet import PreActResNet18
from .utils import get_loaders, get_train_loaders_by_class, post_train

pretrained_model_path = os.path.join('.', 'pretrained_models', 'cifar_model_weights_30_epochs.pth')

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


class DummyArgs:
    def __init__(self):
        self.data_dir = '../cifar-data'
        self.mixup = False
        self.pt_data = 'ori_neigh'
        self.pt_method = 'adv'
        self.pt_iter = 50
        self.rs_neigh = False
        self.blackbox = False

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
        self.transform = transforms.Compose([
            transforms.Normalize(cifar10_mean, cifar10_std)
        ])

        if args is None:
            args = DummyArgs()
        self.args = args

        self.train_loader, _ = get_loaders(self.args.data_dir, batch_size=128)
        self.train_loaders_by_class = get_train_loaders_by_class(self.args.data_dir, batch_size=128)

        self.post_model = None

    # def update_post_model(self, images):
    #     sample_images = images[0, :, :, :].unsqueeze(0)
    #     sample_images = self.transform(sample_images)
    #     del self.post_model
    #     post_model, original_class, neighbour_class, loss_list, acc_list, neighbour_delta = \
    #         post_train(self.model, sample_images, self.train_loader, self.train_loaders_by_class, self.args)
    #     self.post_model = post_model

    def forward(self, images):
        images = self.transform(images)
        sample_images = images[0, :, :, :].unsqueeze(0)
        post_model, original_class, neighbour_class, loss_list, acc_list, neighbour_delta = \
            post_train(self.model, sample_images, self.train_loader, self.train_loaders_by_class, self.args)
        return post_model(images)
        # return self.model(images)