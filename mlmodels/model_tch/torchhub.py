# -*- coding: utf-8 -*-
"""
https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/facebookresearch_pytorch-gan-zoo_pgan.ipynb

https://github.com/pytorch/pytorch/blob/98362d11ffe81ca48748f6b0e1e417cb81ba5998/torch/hub.py#L330


        the following models only: alexnet, densenet121, densenet169, densenet201,\
        densenet161, inception_v3, resnet18, resnet34, resnet50, resnet101, resnet152,\
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, squeezenet1_0,\
        squeezenet1_1, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,\
        googlenet, shufflenet_v2_x0_5, shufflenet_v2_x1_0, mobilenet_v2"


        assert _model in ['alexnet', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 
        'inception_v3', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
        'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
        'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn',
        'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'googlenet', 'shufflenet_v2_x0_5', 
        'shufflenet_v2_x1_0', 'mobilenet_v2'],\
        "Pretrained models are available for \
        the following models only: alexnet, densenet121, densenet169, densenet201,\
        densenet161, inception_v3, resnet18, resnet34, resnet50, resnet101, resnet152,\
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, squeezenet1_0,\
        squeezenet1_1, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,\
        googlenet, shufflenet_v2_x0_5, shufflenet_v2_x1_0, mobilenet_v2"

"""
import os, json
import copy


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import hub 

# import mlmodels.models as M


from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, path_norm_dict
MODEL_URI = get_model_uri(__file__)

# CV datasets come in various formats, we should write a dataloader for each dataset
# I assume that the dataloader (itrator) will be ready and imported from another file

###########################################################################################################
###########################################################################################################
def _train(m, device, train_itr, criterion, optimizer, epoch, max_epoch, imax=1):
    m.train()
    corrects, train_loss = 0.0,0.0

    for i,batch in enumerate(train_itr):
        if i >= imax: break

        image, target = batch[0], batch[1]
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        logit = m(image)
        
        loss = criterion(logit, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(train_itr)
    train_loss /= size 
    accuracy = 100.0 * corrects/size
  
    return train_loss, accuracy
    
def _valid(m, device, test_itr, criterion, imax=1):
    m.eval()
    corrects, test_loss = 0.0,0.0
    for i,batch in enumerate(test_itr):
        if i >= imax: break
        
        image, target = batch[0], batch[1]
        image, target = image.to(device), target.to(device)
        
        logit = m(image)
        loss = criterion(logit, target)

        
        test_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(test_itr)
    test_loss /= size 
    accuracy = 100.0 * corrects/size
    
    return test_loss, accuracy

def _get_device():
    # use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_config_file():
    return path_norm('config/model_tch/Imagecnn.json')


def get_dataset_mnist_torch(data_pars):
    train_loader = torch.utils.data.DataLoader( datasets.MNIST(data_pars['data_path'], train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=data_pars['train_batch_size'], shuffle=True)


    valid_loader = torch.utils.data.DataLoader( datasets.MNIST(data_pars['data_path'], train=False,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=data_pars['test_batch_size'], shuffle=True)
    return train_loader, valid_loader  



def get_dataset_torch(data_pars):
    import importlib

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






###########################################################################################################
###########################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, out_pars=None):
        self.model_pars = copy.deepcopy(model_pars)  
        self.compute_pars = copy.deepcopy(compute_pars)
        m = model_pars 

        ### Model Structure        ################################
        if model_pars is None :
            self.model = None
            return None

        #### Progressive GAN       ################################
        if m['repo_uri'] == 'facebookresearch/pytorch_GAN_zoo:hub' :
           #'DCGAN',
           self.model = torch.hub.load(m['repo_uri'],
                                       m.get('model', 'PGAN'), 
                                       model_name = m.get('model_name', 'celebAHQ-512'),
                                       pretrained = bool( m.get('pretrained', True)), 
                                       useGPU     = compute_pars.get('use_gpu', _get_device()) )
           return None
        

        #### Other CNN models    ################################
        num_classes = m['num_classes']
        _model      = m['model']
        self.model  = hub.load( m['repo_uri'], _model, 
                                # model_name = m.get("model_name", m['model']),
                                pretrained = bool( m.get('pretrained', True)),
                                # useGPU     = m.get('use_gpu',False)
                              ) 

        if num_classes != 1000:
            fc_in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(fc_in_features, num_classes)




def get_params(param_pars=None, **kw):
    pp          = param_pars
    choice      = pp['choice']
    config_mode = pp['config_mode']
    data_path   = pp['data_path']

    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]

        ####Normalize path  : add /models/dataset/
        cf['data_pars'] = path_norm_dict(cf['data_pars'])
        cf['out_pars']  = path_norm_dict(cf['out_pars'])

        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")





def get_dataset(data_pars=None, **kw):

    if data_pars['dataset'] == 'MNIST':
        train_loader, valid_loader  = get_dataset_mnist_torch(data_pars)
        return train_loader, valid_loader  

    elif data_pars['dataset'] :
        train_loader, valid_loader  = get_dataset_torch(data_pars)
        return train_loader, valid_loader  

    else:
        raise Exception("Dataloader not implemented")
        exit



def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    model0        = model.model
    lr            = compute_pars['learning_rate']
    epochs        = compute_pars["epochs"]
    criterion     = nn.CrossEntropyLoss()
    device        = _get_device()
    model0.to(device)
    train_loss    = []
    train_acc     = []
    test_loss     = []
    test_acc      = []
    best_test_acc = -1

    optimizer     = optim.Adam(model0.parameters(), lr=lr)
    train_iter, valid_iter = get_dataset(data_pars)

    imax_train = compute_pars.get('max_batch_sample', len(train_iter) )
    imax_valid = compute_pars.get('max_batch_sample', len(valid_iter) )

    os.makedirs(out_pars["checkpointdir"], exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        #train loss
        tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
        print( f'Train Epoch: {epoch} \t Loss: {tr_loss} \t Accuracy: {tr_acc}')


        ts_loss, ts_acc = _valid(model0, device, valid_iter, criterion, imax=imax_valid)
        print( f'Train Epoch: {epoch} \t Loss: {ts_loss} \t Accuracy: {ts_acc}')

        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            #save paras(snapshot)
            print( f"model saves at {best_test_acc} accuracy")
            torch.save(model0.state_dict(), os.path.join(out_pars["checkpointdir"],  "best_accuracy"))

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(ts_loss)
        test_acc.append(ts_acc)

    model.model = model0
    return model, None


def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None, imax = 1, return_ytrue=1):
    ###### Progressive GAN
    if model.model_pars['repo_uri'] == 'facebookresearch/pytorch_GAN_zoo:hub' :
        model0     = model.model     
        num_images = compute_pars.get('num_images', 4)
        noise, _   = model0.buildNoiseData(num_images)
        with torch.no_grad():
            generated_images = model0.test(noise)

        # let's plot these images using torchvision and matplotlib
        import matplotlib.pyplot as plt
        import torchvision
        grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        # plt.show()

        os.makedirs(out_pars['path'], exist_ok=True)
        plt.savefig(out_pars['path'] + "/img_01.png")
        os.system("ls " + out_pars['path'])
        return 0
   

    ######  CNN models
    import numpy as np
    from mlmodels.metrics import metrics_eval
    device = _get_device()
    model = model.model
    _, test_iter = get_dataset(data_pars=data_pars)

    # test_iter = get_dataset(data_pars, out_pars)
    y_pred = []
    y_true = []
    for i,batch in enumerate(test_iter):
        if i >= imax: break
        image, target = batch[0], batch[1]
        image = image.to(device)
        logit = model(image)
        predictions = torch.max(logit,1)[1].cpu().numpy()
        y_pred.append(predictions)
        y_true.append(target)
    y_pred = np.vstack(y_pred)[0]
    y_true = np.vstack(y_true)[0]

    return y_pred, y_true  if return_ytrue else y_pred


def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None):
    pass


def save(model, session=None, save_pars=None):
    from mlmodels.util import save_tch
    save_tch(model=model, save_pars=save_pars)


def load(load_pars):
    from mlmodels.util import load_tch
    return load_tch(load_pars)



###########################################################################################################
###########################################################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(  data_pars, out_pars )

    log("#### Loading dataset   #############################################")
    xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    ypred = predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save_pars = { "path": out_pars["path"]  }
    save(model=model, save_pars=save_pars)
    model2 = load( save_pars )
    ypred = predict(model2, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(model2)



def test2(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(  data_pars, out_pars )

    log("#### Loading dataset   #############################################")
    #xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    #model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    #metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    #print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save_pars = { "path": out_pars["path"]  }
    save(model=model, save_pars=save_pars)
    model2 = load( save_pars )
    ypred = predict(model2, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(model2)


if __name__ == "__main__":
    test(data_path="model_tch/torchhub_cnn.json", pars_choice="json", config_mode="test")


    test2(data_path="model_tch/torchhub_pgan.json", pars_choice="json", config_mode="test")







