""""
Related to data procesisng

TO DO :
   Normalize datasetloader and embedding loading


1) embeddings can be trainable or fixed  : True
2) embedding are model data, not not split train/test 







"""
import os
from pathlib import Path
import pandas as pd, numpy as np


from mlmodels.util import path_norm, log

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
    ##### Pandas CSV case : Custom MLMODELS One
    "dataset"        : "mlmodels.preprocess.generic:pandasDataset"

    ##### External File processor :
    "dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"

      Absolute drive path
     "MyFolder/mlmodels/preprocess/generic.py:pandasDataset"


  """  
  import importlib, sys
  from pathlib import Path
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
        #log(path_parent)

        #### import Absilute Path model_tf.1_lstm
        model_name   = Path(package).stem  # remove .py
        package_name = str(Path(package).parts[-2]) + "." + str(model_name)
        #log(package_name, model_name)
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


    ##### MNIST case : TorchVison TorchText Pre-Built
    "dataset"       : "torchvision.datasets:MNIST"
    "transform_uri" : "mlmodels.preprocess.image:torch_transform_mnist"


    ##### Pandas CSV case : Custom MLMODELS One
    "dataset"        : "mlmodels.preprocess.generic:pandasDataset"
    "transform_uri"  : "mlmodels.preprocess.text:torch_fillna"


    ##### External File processor :
    "dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
    "transform_uri"  : "MyFolder/preprocess/myfile.py:torch_fillna"


    """
    from torch.utils.data import DataLoader
    d = data_pars

    transform = None
    if  len(data_pars.get("transform_uri", ""))  > 1 :
       transform = load_function( d.get("transform_uri", "mlmodels.preprocess.image:torch_transform_mnist" ))()

    #### from mlmodels.preprocess.image import pandasDataset
    dset = load_function(d.get("dataset", "torchvision.datasets:MNIST") ) 


    if d.get('train_path') and  d.get('test_path') :
        ###### Custom Build Dataset   ####################################################
        dset_inst    = dset(d['train_path'], train=True, download=True, transform= transform, data_pars=data_pars)
        train_loader = DataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))
        
        dset_inst    = dset(d['test_path'], train=False, download=False, transform= transform, data_pars=data_pars)
        valid_loader = DataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))


    else :
        ###### Pre Built Dataset available  #############################################
        dset_inst    = dset(d['data_path'], train=True, download=True, transform= transform)
        train_loader = DataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))
        
        dset_inst    = dset(d['data_path'], train=False, download=False, transform= transform)
        valid_loader = DataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))


    return train_loader, valid_loader  







####Not Yet tested
def get_dataset_keras(data_pars):
    """"
    #### Write someple
    from mlmodels.preprocess.keras_dataloader.dataloader import DataGenerator as kerasDataloader
    from mlmodels.preprocess.keras_dataloader.dataset import Dataset as kerasDataset

    class TensorDataset(kerasDataset):

        def __getitem__(self, index):
            # time.sleep(np.random.randint(1, 3))
            return np.random.rand(3), np.array([index])

        def __len__(self):
            return 100
            
    #model = Sequential()
    #model.add(Dense(units=4, input_dim=3))
    #model.add(Dense(units=1))
    #model.compile('adam', loss='mse')

    data_loader = kerasDataloader(TensorDataset(), batch_size=20, num_workers=0)

    return data_loader
    # model.fit_generator(generator=data_loader, epochs=1, verbose=1)


    ##### MNIST case : TorchVison TorchText Pre-Built
    "dataset"       : "torchvision.datasets:MNIST"
    "transform_uri" : "mlmodels.preprocess.image:torch_transform_mnist"


    ##### Pandas CSV case : Custom MLMODELS One
    "dataset"        : "mlmodels.preprocess.generic:pandasDataset"
    "transform_uri"  : "mlmodels.preprocess.text:torch_fillna"


    ##### External File processor :
    "dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
    "transform_uri"  : "MyFolder/preprocess/myfile.py:torch_fillna"


    """
    from mlmodels.preprocess.keras_dataloader.dataloader import DataGenerator as kerasDataLoader
    d = data_pars

    transform = None
    if  len(data_pars.get("transform_uri", ""))  > 1 :
       transform = load_function( d.get("transform_uri", "mlmodels.preprocess.image:keras_transform_mnist" ))()

    #### from mlmodels.preprocess.image import pandasDataset
    dset = load_function(d.get("dataset", "mlmodels.preprocess.datasets:MNIST") ) 


    if d.get('train_path') and  d.get('test_path') :
        ###### Custom Build Dataset   ####################################################
        dset_inst    = dset(d['train_path'], train=True, download=True, transform= transform, data_pars=data_pars)
        train_loader = kerasDataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))
        
        dset_inst    = dset(d['test_path'], train=False, download=False, transform= transform, data_pars=data_pars)
        valid_loader = kerasDataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))


    else :
        ###### Pre Built Dataset available  #############################################
        dset_inst    = dset(d['data_path'], train=True, download=True, transform= transform, data_pars=data_pars)
        train_loader = kerasDataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))
        
        dset_inst    = dset(d['data_path'], train=False, download=False, transform= transform, data_pars=data_pars)
        valid_loader = kerasDataLoader( dset_inst, batch_size=d['train_batch_size'], shuffle= d.get('shuffle', True))


    return train_loader, valid_loader  





def get_model_embedding(model_pars, data_pars):
    """"
      Mostly Embedding data, it can be external data used in the model.

      INDEPENDANT OF Framework BUT Follows PyTorch Logic

    ##### MNIST case : TorchVison TorchText Pre-Built
    "dataset"       : "torchvision.datasets:MNIST"
    "transform_uri" : "mlmodels.preprocess.image:torch_transform_mnist"


    ##### Pandas CSV case : Custom MLMODELS One
    "dataset"        : "mlmodels.preprocess.generic:pandasDataset"
    "transform_uri"  : "mlmodels.preprocess.text:torch_fillna"


    ##### External File processor :
    "dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
    "transform_uri"  : "MyFolder/preprocess/myfile.py:torch_fillna"


    """
    d = model_pars

    ### Embedding Transformer
    transform = None
    if  len(data_pars.get("embedding_transform_uri", ""))  > 1 :
       transform = load_function( d.get("embedding_transform_uri", "mlmodels.preprocess.text:torch_transform_glove" ))()


    #### from mlmodels.preprocess.text import embeddingLoader
    dset = load_function(d.get("embedding_dataset", "torchtext.embedding:glove") )

    data = None
    if len(d.get('embedding_path', "")) > 1 :
        ###### Custom Build Dataset   ####################################################
        data    = dset(d['embedding_path'], train=True, download=True, transform= transform, model_pars=model_pars, data_pars=data_pars)
        

    else :
        ###### Pre Built Dataset available  #############################################
        data    = dset(d['embedding_path'], train=True, download=True, transform= transform)


    return data





def text_create_tabular_dataset(path_train, path_valid,   lang='en', pretrained_emb='glove.6B.300d'):
    import spacy
    import torchtext
    from torchtext.data import Field
    from torchtext.data import TabularDataset
    from torchtext.vocab import GloVe
    from torchtext.data import Iterator, BucketIterator
    import torchtext.datasets
    from time import sleep
    import re


    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()


    #### Tokenizer  ################################################
    disable = [ 'tagger', 'parser', 'ner', 'textcat'
        'entity_ruler', 'sentencizer', 
        'merge_noun_chunks', 'merge_entities',
        'merge_subtokens']
    try :
      spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)

    except :
       #### Very hacky to get Glove Data 
       log( f"Download {lang}")
       os.system( f"python -m spacy download {lang}")
       sleep(60)
       spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  

    def tokenizer(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]


    # Creating field for text and label  ###########################
    TEXT  = Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = Field(sequential=False)


    log('Preprocessing the text...')
    TEXT.preprocessing = torchtext.data.Pipeline(clean_str)



    log('Creating tabular datasets...It might take a while to finish!')
    train_datafield = [('text', TEXT), ('label', LABEL)]
    tabular_train   = TabularDataset(path=path_train, format='csv', skip_header=True, fields=train_datafield)

    valid_datafield = [('text', TEXT), ('label', LABEL)]
    tabular_valid   = TabularDataset(path=path_valid, format='csv', skip_header=True, fields=valid_datafield)


    log('Building vocaulary...')
    TEXT.build_vocab(tabular_train, vectors=pretrained_emb)
    LABEL.build_vocab(tabular_train)


    return tabular_train, tabular_valid, TEXT.vocab






from PIL import Image
# from matplotlib import cm
# im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))
# import torch
# torch.from_numpy(X).float()
class NumpyDataset(Dataset):
    """
    Defines a dataset composed of Features and labels

    Attributes:
        data_pars{
            data_path: the folder path that it cotains numpy files
            filename: the name of numpy file 
            features_key: the key of features ex: n = {"features_name":"data}; n["features_name"]
            classes_key: the key of classes
            transforms: operation you wanna apply on image

            example:
                dataset_pars = {'data_path': 'mlmodels/dataset/vision/cifar10/', 
                                    'filename':'cifar10_train.npz',
                                    'features_name':'X',
                                    'classes_name':'y',
                                    'transforms':transforms.Compose([transforms.Resize(255), 
                                                        transforms.CenterCrop(224),  
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(), 
                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) }
        }        
    """

    def __init__(self, root="", train=True, transform=None, target_transform=None,
                 download=False, data_pars=None):

        # path = data_pars['data_path']
        # file_name = data_pars['filename']
        # data      = np.load(os.path.join("mlmodels", path, file_name))
        # self.features = data[data_pars['features_key']]
        # self.classes = data[data_pars['classes_key']]
        # self.transforms = data_pars['transform_uri']
        if download:
            tf_dataset_download(data_pars)
        
        self.target_transform = target_transform
        self.transform  = transform
        self.to_image   = data_pars.get('to_image', 1)


        file_name       = data_pars['dataset_train_file_name'] if train else data_pars['dataset_test_file_name']
        data            = np.load( path_norm( file_name))
        self.features   = data[data_pars['dataset_features_key']]
        self.classes    = data[data_pars['dataset_classes_key']]


    def __getitem__(self, index):

        X, y = self.features[index], self.classes[index]
        # X =  np.stack((X, X, X)) # gray to rgb 64x64 to 3x64x64

        if self.to_image :
            X = Image.fromarray(np.uint8(X))

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y

    def __len__(self):
        return len(self.features)


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





def create_kerasDataloader():
    """
    keras dataloader
    DataLoader for keras

    Usage example
    from mlmodels.preprocess.keras_dataloader.dataloader import DataGenerator as kerasDataloader
    from mlmodels.preprocess.keras_dataloader.dataset import Dataset as kerasDataset


    class TensorDataset(kerasDataset):

        def __getitem__(self, index):
            # time.sleep(np.random.randint(1, 3))
            return np.random.rand(3), np.array([index])

        def __len__(self):
            return 100
            
    model = Sequential()
    model.add(Dense(units=4, input_dim=3))
    model.add(Dense(units=1))
    model.compile('adam', loss='mse')

    data_loader = kerasDataGenerator(TensorDataset(), batch_size=20, num_workers=0)

    model.fit_generator(generator=data_loader, epochs=1, verbose=1)
    """
    #### Write someple
    from mlmodels.preprocess.keras_dataloader.dataloader import DataGenerator as kerasDataloader
    from mlmodels.preprocess.keras_dataloader.dataset import Dataset as kerasDataset



    class TensorDataset(kerasDataset):

        def __getitem__(self, index):
            # time.sleep(np.random.randint(1, 3))
            return np.random.rand(3), np.array([index])

        def __len__(self):
            return 100
            
    #model = Sequential()
    #model.add(Dense(units=4, input_dim=3))
    #model.add(Dense(units=1))
    #model.compile('adam', loss='mse')

    data_loader = kerasDataloader(TensorDataset(), batch_size=20, num_workers=0)

    return data_loader
    # model.fit_generator(generator=data_loader, epochs=1, verbose=1)




###############################################################################################################
def tf_dataset_download(data_pars):
    """
        Save in numpy compressez format TF Datasets
    
        dataset_pars ={ "dataset_id" : "mnist", "batch_size" : 5000, "n_train": 500, "n_test": 500, 
                            "out_path" : "dataset/vision/mnist2/" }
        tf_dataset_download(dataset_pars)
        
        
        https://www.tensorflow.org/datasets/api_docs/python/tfds
        import tensorflow_datasets as tfds
        import tensorflow as tf
        
        # Here we assume Eager mode is enabled (TF2), but tfds also works in Graph mode.
        log(tfds.list_builders())
        
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

    d          = data_pars
    log( d['dataset'])
    dataset_id = d['dataset'].split(":")[-1].lower()


    n_train    = d.get("tfdataset_train_samples", 500)
    n_test     = d.get("tfdataset_test_samples", 50)
    batch_size = d.get("tfdataset_train_batch_size", 10)
    out_path   = path_norm(d['tf_data_path'] )


    name       = dataset_id.replace(".","-")    
    os.makedirs(out_path, exist_ok=True) 
    log("Dataset Name is : ", name)



    train_ds =  tfds.as_numpy( tfds.load(dataset_id, split= f"train[0:{n_train}]") )
    test_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]") )
    # val_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )

    # log("train", train_ds.shape )
    # log("test",  test_ds.shape )

    def get_keys(x):
        if "image" in x.keys() : xkey = "image"
        if "text" in x.keys() : xkey = "text"    
        return xkey

    Xtemp = []
    ytemp = []
    for x in train_ds:
        #log(x)
        xkey =  get_keys(x)
        Xtemp.append(x[xkey])
        ytemp.append(x.get('label'))

    Xtemp = np.array(Xtemp)
    ytemp = np.array(ytemp)
    np.savez_compressed(os.path.join(out_path + f"{name}_train") , X = Xtemp, y = ytemp )    

    Xtemp = []
    ytemp = []
    for x in test_ds:
        #log(x)
        Xtemp.append(x[xkey])
        ytemp.append(x.get('label'))
    Xtemp = np.array(Xtemp)
    ytemp = np.array(ytemp)
    np.savez_compressed(os.path.join(out_path + f"{name}_test"), X = Xtemp, y = ytemp)
        
    log(out_path, os.listdir( out_path ))
        
      


########################################################################################
########################################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")



if __name__ == "__main__":
    test(data_path="model_tch/file.json", pars_choice="json", config_mode="test")











'''
class numpyDataset(Dataset):
    """
    Defines a dataset composed of sentiment text and labels
    Attributes:
        X: numpy tensor of the path
        y: numpy for labels
        sample_weights(ndarray, shape(len(labels),)): An array with each sample_weight[i] as the weight of the ith sample
        data (list[int, [int]]): The data in the set


    """
   
    def __init__(self,root="", train=True, transform=None, target_transform=None,
                 download=False, data_pars=None, ):
        import torch
        import numpy as np
        self.data_pars        = data_pars
        self.transform        = transform
        self.target_transform = target_transform
        self.download         = download
        d = data_pars


        path = d['train_path'] if train else d['test_path']
        #filename = d['X_filename'], d['y_filename']
        #colX =d['colX']


        # df = torch.load(os.path.join(path, filename))
        X      = np.load(os.path.join(path, d['X_filename']))
        labels = np.load(os.path.join(path, d['y_filename'] )) 
        # self.X = X
        # self.labels = labels


        #### Split  ####################
        #X = df[ colX ]
        #labels = df[ d["coly"] ]


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

'''
