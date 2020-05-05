
import os
import sys
import numpy as np, pandas as pd


from mlmodels.util import os_package_root_path, path_norm




def tf_dataset(dataset_pars):
    """
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

    d          = dataset_pars
    dataset_id = d['dataset_id']
    batch_size = d.get('batch_size', -1)  # -1 neans all the dataset
    n_train    = d.get("n_train", 500)
    n_test     = d.get("n_test", 500)
    out_path   = path_norm(d['out_path'] )
    name       = dataset_id.replace(".","-")    
    os.makedirs(out_path, exist_ok=True) 


    train_ds =  tfds.as_numpy( tfds.load(dataset_id, split= f"train[0:{n_train}]", batch_size=batch_size) )
    test_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )

    # test_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )


    
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
        
                



    
"""

Dataset from pytorch Vision
https://pytorch.org/docs/master/torchvision/datasets.html#fashion-mnist



Dataset from Torch Text
https://pytorch.org/text/datasets.html



All dataset
https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research




Time series
https://gluon-ts.mxnet.io/api/gluonts/gluonts.dataset.repository.datasets.html


https://github.com/zalandoresearch/fashion-mnist





####LudWig

https://blog.dominodatalab.com/a-practitioners-guide-to-deep-learning-with-ludwig/






class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transforms=None):
        
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transforms = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28]) 
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28,28).astype('uint8')
  # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)
        

if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])
    custom_mnist_from_csv = \
        CustomDatasetFromCSV('../data/mnist_in_csv.csv', 28, 28, transformations)
        
"""





def download_data(data_pars):
  """

   dataset/

   Prefix based :
      repo::
      dropbox::

   import_data 

   preprocess_data

  download_data({"from_path" :  "tabular",  
                        "out_path" :  path_norm("ztest/dataset/text/") } )

  Open URL
     https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAoFh0aO9RqwwROksGgasIha?dl=0


  """
  from cli_code.cli_download import Downloader

  folder = data_pars['from_path']  # dataset/text/

  urlmap = {
     "text" :    "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AADHrhC7rLkd42_CEqK6A9oYa/dataset/text?dl=1&subfolder_nav_tracking=1"
     ,"tabular" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAxZkJTGSumLADzj3B5wbA0a/dataset/tabular?dl=1&subfolder_nav_tracking=1"
     ,"pretrained" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AACL3LHW1USWrvsV5hipw27ia/model_pretrained?dl=1&subfolder_nav_tracking=1"

     ,"vision" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAM4k7rQrkjBo09YudYV-6Ca/dataset/vision?dl=1&subfolder_nav_tracking=1"
     ,"recommender": "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AABIb2JjQ6aQHwfq5CU0ypHOa/dataset/recommender?dl=1&subfolder_nav_tracking=1"

  }

  url = urlmap[folder]

  #prefix = "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/"
  #url= f"{prefix}/AADHrhC7rLkd42_CEqK6A9oYa/{folder}?dl=1&subfolder_nav_tracking=1"

  out_path = data_pars['out_path']

  zipname = folder.split("/")[0]


  os.makedirs(out_path, exist_ok=True)
  downloader = Downloader(url)
  downloader.download(out_path)

  import zipfile
  with zipfile.ZipFile( out_path + "/" + zipname + ".zip" ,"r") as zip_ref:
      zip_ref.extractall(out_path)




####################################################################################
def import_data_tch(name="", mode="train", node_id=0, data_folder_root=""):
    import torch.utils.data.distributed
    from torchvision import datasets, transforms

    if name == "mnist" :
        data_folder = os.path.join( data_folder_root,  "data-%d" % node_id)
        dataset = datasets.MNIST(
            data_folder,
            train=True if mode =="train" else False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return dataset


def import_data_fromfile(**kw):
   """
       data_pars["data_path"]
   
   """ 
   import pandas as pd
   import numpy as np

   m = kw
   if m.get("uri_type") in ["pickle", "pandas_pickle" ]:
       df = pd.read_pickle(m["data_path"])
       return df


   if m.get("uri_type") in ["csv", "pandas_csv" ]:
       df = pd.read_csv(m["data_path"])
       return df


   if m.get("uri_type") in [ "dask" ]:
       df = pd.read_csv(m["data_path"])
       return df


   if ".npz" in m['data_path'] :
       arr = import_to_numpy(data_pars, mode=mode, **kw)
       return arr


   if ".csv" in m['data_path'] or ".txt" in m['data_path']  :
       df = import_to_pandas(data_pars, mode=mode, **kw)
       return df


   if ".pkl" in m['data_path']   :
       df = pd.read_pickle(m["data_path"], **kw)
       return df




def import_data_dask(**kw):
  extension = kw['path'].split(".")[-1]

  if kw.get("use_dask", False):
    import dask.dataframe as dd
    if extension in [".csv", ".txt"]: 
       df = dd.read_csv(kw["data_path"])
    elif extension in [".pkl"]: 
       df = dd.read_pickle(kw["data_path"])
    elif extension in [".npz"]: 
       df = dd.read_pickle(kw["data_path"])
    else: raise Exception(f"Not support extension {extension}")
  return df






def import_data():
  def sent_generator(TRAIN_DATA_FILE, chunksize):
      import pandas as pd
      reader = pd.read_csv(TRAIN_DATA_FILE, chunksize=chunksize, iterator=True)
      for df in reader:
          val3  = df.iloc[:, 3:4].values.tolist()
          val4  = df.iloc[:, 4:5].values.tolist()
          flat3 = [item for sublist in val3 for item in sublist]
          flat4 = [str(item) for sublist in val4 for item in sublist]
          texts = []
          texts.extend(flat3[:])
          texts.extend(flat4[:])

          sequences  = model.tokenizer.texts_to_sequences(texts)
          data_train = pad_sequences(sequences, maxlen=data_pars["MAX_SEQUENCE_LENGTH"])
          yield [data_train, data_train]

  model.model.fit(sent_generator(data_pars["train_data_path"], batch_size / 2),
                  epochs          = epochs,
                  steps_per_epoch = n_steps,
                  validation_data = (data_pars["data_1_val"], data_pars["data_1_val"]))






def get_dataset(data_pars) : 
  """
    path:
    is_local  : Local to the repo 
    data_type:
    train : 1/0
    data_source :  ams
  """
  dd = data_pars

  if not d.get('is_local') is None :
      dd['path'] = os_package_root_path(__file__, 0, dd['path'] )


  if dd['train'] :
     df = pd.read_csv(path) 



     ### Donwload from external


     ## Get from csv, local 


     ## Get from csv, external


     ### Get from external tool


