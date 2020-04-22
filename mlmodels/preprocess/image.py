def torchvision_dataset_MNIST_load(path, **args):
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


def wrap_torch_datasets(sets, args_list = None, **args):
    if not isinstance(sets,list) and not isinstance(sets,tuple):
        sets = [sets]
    import torch
    if args_list is None:
        return [torch.utils.data.DataLoader(x,**args) for x in sets]
    return [torch.utils.data.DataLoader(x,**a,**args) for a,x in zip(args_list,sets)]







def torch_transform_mnist():
    from torchvision import datasets, transforms
    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform






def get_dataset_torch(data_pars):
    import importlib
    import torch

    if  data_pars["transform"]  :
       transform = getattr(importlib.import_module("mlmodels.preprocess.image"), data_pars.get("transform", "torch_transform_mnist" ))
    else :
       transform = None

    
    dataset_module =  data_pars.get('dataset_module', "torchvision.datasets")   
    dset = getattr(importlib.import_module(dataset_module), data_pars["dataset"])

    train_loader = torch.utils.data.DataLoader( dset(data_pars['data_path'], train=True, download=True, transform= transform),
                                                batch_size=data_pars['train_batch_size'], shuffle=True)
    
    valid_loader = torch.utils.data.DataLoader( dset(data_pars['data_path'], train=False, download=True, transform= transform),
                                                batch_size=data_pars['train_batch_size'], shuffle=True)

    return train_loader, valid_loader  






















