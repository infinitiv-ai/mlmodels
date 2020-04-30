""""
Related to images

Examples :
https://www.programcreek.com/python/example/104832/torchvision.transforms.Compose

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


def torch_transform_data_augment(fixed_scale = 256, train = False):
    """
    Options:
    1.RandomCrop
    2.CenterCrop
    3.RandomHorizontalFlip
    4.Normalize
    5.ToTensor
    6.FixedResize
    7.RandomRotate
    """
    from torchvision import  transforms
    size = fixed_scale - 2
    rotate_prob = 0.5

    transform_list = [] 
    #transform_list.append(FixedResize(size = (fixed_scale, fixed_scale)))
    transform_list.append(transforms.RandomSized(fixed_scale))
    transform_list.append(transforms.RandomRotate(rotate_prob))
    transform_list.append(transforms.RandomHorizontalFlip())
    #transform_list.append(Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    transform_list.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list) 




def torch_transform_generic(fixed_scale = 256, train = False):
    from torchvision import  transforms
    size = fixed_scale - 2
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([fixed_scale, fixed_scale]),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        }
    return transform['train' if train else 'test'] 




