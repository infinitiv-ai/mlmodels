""""
Related to images

"""
import os
import pandas as pd, numpy as np


from mlmodels.util import path_norm


from mlmodels.preprocess.generic import get_dataset_torch, torch_datasets_wrapper, load_function
###############################################################################################################









###############################################################################################################
############### Custom Code ###################################################################################
def torch_transform_mnist():
    from torchvision import datasets, transforms
    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform



def torchvision_dataset_MNIST_load(path, **args):
    ### only used in Refactoring part
    from torchvision import datasets, transforms
    train_dataset = datasets.MNIST(path, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    valid_dataset = datasets.MNIST(path, train=False,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    return train_dataset, valid_dataset  









