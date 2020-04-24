""""

Related to data procesisng



"""
import os
from pathlib import Path
import pandas as pd, numpy as np


from mlmodels.util import path_norm

from torch.utils.data import Dataset


###############################################################################################################
###############################################################################################################
def torch_datasets_wrapper(sets, args_list = None, **args):
    if not isinstance(sets,list) and not isinstance(sets,tuple):
        sets = [sets]
    import torch
    if args_list is None:
        return [torch.utils.data.DataLoader(x,**args) for x in sets]
    return [torch.utils.data.DataLoader(x,**a,**args) for a,x in zip(args_list,sets)]



def load_function(uri_name="path_norm"):
  """
     "mlmodels.preprocess.generic:pandasDataset"

      Absolute drive path
     "MyFolder/mlmodels/preprocess/generic.py:pandasDataset"


  """  
  import importlib, sys
  pkg = uri_name.split(":")
  package, name = pkg[0], pkg[1]

  try:
    #### Import from package mlmodels sub-folder
    return  getattr(importlib.import_module(package), name)

  except Exception as e1:
    try:
        ### Add Folder to Path and Load absoluate path module
        path_parent = str(Path(package).parent.parent.absolute())
        sys.path.append(path_parent)
        #print(path_parent)

        #### import Absilute Path model_tf.1_lstm
        model_name   = Path(package).stem  # remove .py
        package_name = str(Path(package).parts[-2]) + "." + str(model_name)
        #print(package_name, model_name)
        return  getattr(importlib.import_module(package_name), name)

    except Exception as e2:
        raise NameError(f"Module {pkg} notfound, {e1}, {e2}")




def get_dataset_torch(data_pars):
    """"
      torchvison.datasets
         MNIST Fashion-MNIST KMNIST EMNIST QMNIST  FakeData COCO Captions Detection LSUN ImageFolder DatasetFolder 
         ImageNet CIFAR STL10 SVHN PhotoTour SBU Flickr VOC Cityscapes SBD USPS Kinetics-400 HMDB51 UCF101 CelebA

      torchtext.datasets
         Sentiment Analysis:    SST IMDb Question Classification TREC Entailment SNLI MultiNLI 
         Language Modeling:     WikiText-2 WikiText103  PennTreebank 
         Machine Translation :  Multi30k IWSLT WMT14 
         Sequence Tagging    :  UDPOS CoNLL2000Chunking 
         Question Answering  :  BABI20


    ##### MNIST case 
    "dataset"       : "torchvision.datasets:MNIST"
    "transform_uri" : "mlmodels.preprocess.image:torch_transform_mnist"


    ##### Pandas CSV case
    "dataset"        : "mlmodels.preprocess.generic:pandasDataset"
    "transform_uri"  : "mlmodels.preprocess.text:torch_fillna"


    """
    import torch
    from torch.utils.data import DataLoader
    d = data_pars

    transform = None
    if  data_pars.get("transform_uri")   :
       transform = load_function( d.get("transform_uri", "mlmodels.preprocess.image:torch_transform_mnist" ))()


    #### from mlmodels.preprocess.image import pandasDataset
    dset = load_function(d.get("dataset", "torchvision.datasets:MNIST") )


    if d.get('train_path') and  d.get('test_path') :
        ###### Custom Build Dataset   ####################################################
        dset_inst    = dset(d['train_path'], train=True, download=True, transform= transform, data_pars=data_pars)
        train_loader = DataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))
        
        dset_inst    = dset(d['test_path'], train=False, download=True, transform= transform, data_pars=data_pars)
        valid_loader = DataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))


    else :
        ###### Pre Built Dataset available  #############################################
        dset_inst    = dset(d['data_path'], train=True, download=True, transform= transform)
        train_loader = DataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))
        
        dset_inst    = dset(d['data_path'], train=False, download=True, transform= transform)
        valid_loader = DataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))


    return train_loader, valid_loader  




class pandasDataset(Dataset):
    """
    Defines a dataset composed of sentiment text and labels
    Attributes:
        df (Dataframe): Dataframe of the CSV from teh path
        sample_weights(ndarray, shape(len(labels),)): An array with each sample_weight[i] as the weight of the ith sample
        data (list[int, [int]]): The data in the set
    """
   
    def __init__(self,root="", train=True, transform=None, target_transform=None,
                 download=False, data_pars=None, ):
        import torch
        self.data_pars        = data_pars
        self.transform        = transform
        self.target_transform = target_transform
        self.download         = download
        d = data_pars


        path = d['train_path'] if train else d['test_path']
        filename = d['filename']
        colX =d['colX']


        # df = torch.load(os.path.join(path, filename))
        df = pd.read_csv(os.path.join(path, filename))
        self.df = df


        #### Split  ####################
        X = df[ colX ]
        labels = df[ d["coly"] ]


        #### Compute sample weights from inverse class frequencies
        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        self.samples_weight = torch.from_numpy(weight[labels])


        #### Data Joining  ############
        self.data = list(zip(X, labels))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        X, target = self.data[index], int(self.targets[index])


        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

    def shuffle(self, random_state=123):
            self._df = self._df.sample(frac=1.0, random_state=random_state)





###############################################################################################################
def tf_dataset(dataset_pars):
    """
        Save in numpy compressez format TF Datasets
    
        dataset_pars ={ "dataset_id" : "mnist", "batch_size" : 5000, "n_train": 500, "n_test": 500, 
                            "out_path" : "dataset/vision/mnist2/" }
        tf_dataset(dataset_pars)
        
        
        https://www.tensorflow.org/datasets/api_docs/python/tfds
        import tensorflow_datasets as tfds
        import tensorflow as tf
        
        # Here we assume Eager mode is enabled (TF2), but tfds also works in Graph mode.
        print(tfds.list_builders())
        
        # Construct a tf.data.Dataset
        ds_train = tfds.load(name="mnist", split="train", shuffle_files=True)
        
        # Build your input pipeline
        ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
        for features in ds_train.take(1):
          image, label = features["image"], features["label"]
          
          
        NumPy Usage with tfds.as_numpy
        train_ds = tfds.load("mnist", split="train")
        train_ds = train_ds.shuffle(1024).batch(128).repeat(5).prefetch(10)
        
        for example in tfds.as_numpy(train_ds):
          numpy_images, numpy_labels = example["image"], example["label"]
        You can also use tfds.as_numpy in conjunction with batch_size=-1 to get the full dataset in NumPy arrays from the returned tf.Tensor object:
        
        train_ds = tfds.load("mnist", split=tfds.Split.TRAIN, batch_size=-1)
        numpy_ds = tfds.as_numpy(train_ds)
        numpy_images, numpy_labels = numpy_ds["image"], numpy_ds["label"]
        
        
        FeaturesDict({
    'identity_attack': tf.float32,
    'insult': tf.float32,
    'obscene': tf.float32,
    'severe_toxicity': tf.float32,
    'sexual_explicit': tf.float32,
    'text': Text(shape=(), dtype=tf.string),
    'threat': tf.float32,
    'toxicity': tf.float32,
})
            
            
    
    """
    import tensorflow_datasets as tfds
    import numpy as np

    d          = dataset_pars
    dataset_id = d['dataset_id']
    batch_size = d.get('batch_size', -1)  # -1 neans all the dataset
    n_train    = d.get("n_train", 500)
    n_test     = d.get("n_test", 500)
    out_path   = path_norm(d['out_path'] )
    name       = dataset_id.replace(".","-")    
    os.makedirs(out_path, exist_ok=True) 


    train_ds = tfds.as_numpy( tfds.load(dataset_id, split= f"train[0:{n_train}]", batch_size=batch_size) )
    test_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )
    # val_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )

    print("train", train_ds.shape )
    print("test",  test_ds.shape )

    def get_keys(x):
       if "image" in x.keys() : xkey = "image"
       if "text" in x.keys() : xkey = "text"    
       return xkey
    
    
    for x in train_ds:
       #print(x)
       xkey =  get_keys(x)
       np.savez_compressed(out_path + f"{name}_train" , X = x[xkey] , y = x.get('label') )
        

    for x in test_ds:
       #print(x)
       np.savez_compressed(out_path + f"{name}_test", X = x[xkey] , y = x.get('label') )
        
    print(out_path, os.listdir( out_path ))
        
      





