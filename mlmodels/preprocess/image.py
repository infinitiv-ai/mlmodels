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