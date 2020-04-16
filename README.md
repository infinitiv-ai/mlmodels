# mlmodels : Model ZOO for Pytorch, Tensorflow, Keras, Gluon, LightGBM, Sklearn models...

- Model ZOO with Lightweight Functional interface to wrap access to Recent and State of Art Deep Learning, ML models and Hyper-Parameter Search, cross platforms such as Tensorflow, Pytorch, Gluon, Keras, sklearn, light-GBM,...

- Logic follows sklearn : fit, predict, transform, metrics, save, load

- Goal is to transform Script/Research code into Re-usable/batch/ code with minimal code change ...

- Why Functional interface instead of pure OOP ?
  Functional reduces the amount of code needed, focus more on the computing part (vs design part),
  a bit easier maintenability for medium size project, good for scientific computing process.


*  Usage, Example :
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/README_usage.md


*  Colab demo for Install :
https://colab.research.google.com/drive/1sYbrXNZh9nTeizS-AuCA8RSu94B_B-RF


## Benefits :

Having a standard framework for both machine learning models and deep learning models, 
allows a step towards automatic Machine Learning. The collection of models, model zoo in Pytorch, Tensorflow, Keras
allows removing dependency on one specific framework, and enable richer possibilities in model benchmarking and re-usage.
Unique and simple interface, zero boilerplate code (!), and recent state of art models/frameworks are the main strength 
of mlmodels.




## Model List :
### Time Series:

Nbeats: 2019, Time Series NNetwork, https://arxiv.org/abs/1905.10437

Amazon Deep AR: 2019, Time Series NNetwork, https://arxiv.org/abs/1905.10437

Facebook Prophet 2017, Time Series prediction,

ARMDN Advanced Time series Prediction : 2019, Associative and Recurrent Mixture Density Networks for time series.

LSTM prediction


### NLP :

Sentence Transformers : 2019, Embedding of full sentences using BERT, https://arxiv.org/pdf/1908.10084.pdf

Transformers Classifier : Using Transformer for Text Classification, https://arxiv.org/abs/1905.05583

TextCNN Pytorch : 2016, Text CNN Classifier, https://arxiv.org/abs/1801.06287

TextCNN Keras : 2016, Text CNN Classifier, https://arxiv.org/abs/1801.06287

DRMM:  Deep Relevance Matching Model for Ad-hoc Retrieval.https://dl.acm.org/doi/pdf/10.1145/2983323.2983769?download=true

DRMMTKS:  Deep Top-K Relevance Matching Model for Ad-hoc Retrieval. 
https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2

ARC-I:  Convolutional Neural Network Architectures for Matching Natural Language Sentences
http://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf

ARC-II:  Convolutional Neural Network Architectures for Matching Natural Language Sentences
http://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf

DSSM:  Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
https://dl.acm.org/doi/pdf/10.1145/2505515.2505665

CDSSM:  Learning Semantic Representations Using Convolutional Neural Networks for Web Search
https://dl.acm.org/doi/pdf/10.1145/2567948.2577348

MatchLSTM: Machine Comprehension Using Match-LSTM and Answer Pointer
https://arxiv.org/pdf/1608.07905

DUET:  Learning to Match Using Local and Distributed Representations of Text for Web Search
https://dl.acm.org/doi/pdf/10.1145/3038912.3052579

KNRM:  End-to-End Neural Ad-hoc Ranking with Kernel Pooling
https://dl.acm.org/doi/pdf/10.1145/3077136.3080809

ConvKNRM:  Convolutional neural networks for soft-matching n-grams in ad-hoc search
https://dl.acm.org/doi/pdf/10.1145/3159652.3159659

ESIM:  Enhanced LSTM for Natural Language Inference
https://arxiv.org/pdf/1609.06038

BiMPM:  Bilateral Multi-Perspective Matching for Natural Language Sentences
https://arxiv.org/pdf/1702.03814

MatchPyramid:  Text Matching as Image Recognition
https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11895/12024

Match-SRNN:  Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN
https://arxiv.org/pdf/1604.04378

aNMM:  aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model
https://dl.acm.org/doi/pdf/10.1145/2983323.2983818

MV-LSTM:  https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11897/12030

DIIN:  Natural Lanuguage Inference Over Interaction Space
https://arxiv.org/pdf/1709.04348

HBMP:  Sentence Embeddings in NLI with Iterative Refinement Encoders
https://www.cambridge.org/core/journals/natural-language-engineering/article/sentence-embeddings-in-nli-with-iterative-refinement-encoders/AC811644D52446E414333B20FEACE00F

### TABULAR :

#### LightGBM

#### AutoML Gluon  :  2020, AutoML in Gluon, MxNet using LightGBM, CatBoost

#### Auto-Keras  :  2020, Automatic Keras model selection


#### All sklearn models :

linear_model.ElasticNet\
linear_model.ElasticNetCV\
linear_model.Lars\
linear_model.LarsCV\
linear_model.Lasso\
linear_model.LassoCV\
linear_model.LassoLars\
linear_model.LassoLarsCV\
linear_model.LassoLarsIC\
linear_model.OrthogonalMatchingPursuit\
linear_model.OrthogonalMatchingPursuitCV\


svm.LinearSVC\
svm.LinearSVR\
svm.NuSVC\
svm.NuSVR\
svm.OneClassSVM\
svm.SVC\
svm.SVR\
svm.l1_min_c\


neighbors.KNeighborsClassifier\
neighbors.KNeighborsRegressor\
neighbors.KNeighborsTransformer\


#### Binary Neural Prediction from tabular data:

A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)             |

Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |

Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |

Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |

DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |

Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |

Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |

Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |

Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |

xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |

AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |

Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)                                                       |

Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |

Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                |

Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                             |

Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)                                                |

FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |




### VISION :


  Vision Models (pre-trained) :  
alexnet: SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
https://arxiv.org/pdf/1602.07360

densenet121: Adversarial Perturbations Prevail in the Y-Channel of the YCbCr Color Space
https://arxiv.org/pdf/2003.00883.pdf

densenet169: Classification of TrashNet Dataset Based on Deep Learning Models
https://ieeexplore.ieee.org/abstract/document/8622212

densenet201: Utilization of DenseNet201 for diagnosis of breast abnormality
https://link.springer.com/article/10.1007/s00138-019-01042-8

densenet161: Automated classification of histopathology images using transfer learning
https://doi.org/10.1016/j.artmed.2019.101743

inception_v3: Menfish Classification Based on Inception_V3 Convolutional Neural Network
https://iopscience.iop.org/article/10.1088/1757-899X/677/5/052099/pdf 

resnet18: Leveraging the VTA-TVM Hardware-Software Stack for FPGA Acceleration of 8-bit ResNet-18 Inference
https://dl.acm.org/doi/pdf/10.1145/3229762.3229766

resnet34: Automated Pavement Crack Segmentation Using Fully Convolutional U-Net with a Pretrained ResNet-34 Encoder
https://arxiv.org/pdf/2001.01912

resnet50: Extremely Large Minibatch SGD: Training ResNet-50 on ImageNet in 15 Minutes
https://arxiv.org/pdf/1711.04325

resnet101: Classification of Cervical MR Images using ResNet101
https://www.ijresm.com/Vol.2_2019/Vol2_Iss6_June19/IJRESM_V2_I6_69.pdf

resnet152: Deep neural networks show an equivalent and often superior performance to dermatologists in onychomycosis diagnosis: Automatic construction of onychomycosis datasets by region-based convolutional deep neural network
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5774804/pdf/pone.0191493.pdf


resnext50_32x4d: Automatic Grading of Individual Knee Osteoarthritis Features in Plain Radiographs using Deep Convolutional Neural Networks
https://arxiv.org/pdf/1907.08020

resnext101_32x8d: DEEP LEARNING BASED PLANT PART DETECTION IN GREENHOUSE SETTINGS
https://efita-org.eu/wp-content/uploads/2020/02/7.-efita25.pdf

wide_resnet50_2: Identiﬁcac¸˜ao de Esp´ecies de ´Arvores por Imagens de Tronco Utilizando Aprendizado de Ma´quina Profundo
http://www.ic.unicamp.br/~reltech/PFG/2019/PFG-19-50.pdf

wide_resnet101_2: Identification of Tree Species by Trunk Images Using Deep Machine Learning
http://www.ic.unicamp.br/~reltech/PFG/2019/PFG-19-50.pdf

squeezenet1_0: Classification of Ice Crystal Habits Observed From Airborne Cloud Particle Imager by Deep Transfer Learning
https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2019EA000636

squeezenet1_1: Benchmarking parts based face processing in-the-wild for gender recognition and head pose estimation
https://doi.org/10.1016/j.patrec.2018.09.023

vgg11: ernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
https://arxiv.org/pdf/1801.05746

vgg13: Convolutional Neural Network for Raindrop Detection
https://ieeexplore.ieee.org/abstract/document/8768613

vgg16: Automatic detection of lumen and media in the IVUS images using U-Net with VGG16 Encoder
https://arxiv.org/pdf/1806.07554

vgg19: A New Transfer Learning Based on VGG-19 Network for Fault Diagnosis
https://ieeexplore.ieee.org/abstract/document/8791884

vgg11_bn:Shifted Spatial-Spectral Convolution for Deep Neural Networks
https://dl.acm.org/doi/pdf/10.1145/3338533.3366575

vgg13_bn: DETOX: A Redundancy-based Framework for Faster and More Robust Gradient Aggregation
http://papers.nips.cc/paper/9220-detox-a-redundancy-based-framework-for-faster-and-more-robust-gradient-aggregation.pdf

vgg16_bn: Partial Convolution based Padding
https://arxiv.org/pdf/1811.11718


vgg19_bn: NeurIPS 2019 Disentanglement Challenge: Improved Disentanglement through Learned Aggregation of Convolutional Feature Maps
https://arxiv.org/pdf/2002.12356


googlenet: On the Performance of GoogLeNet and AlexNet Applied to Sketches
https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12278/11712


shufflenet_v2_x0_5: Exemplar Normalization for Learning Deep Representation
https://arxiv.org/pdf/2003.08761


shufflenet_v2_x1_0: Tree Species Identification by Trunk Images Using Deep Machine Learning
http://www.ic.unicamp.br/~reltech/PFG/2019/PFG-19-50.pdf


mobilenet_v2: MobileNetV2: Inverted Residuals and Linear Bottlenecks
http://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf

A lot more...



......

https://github.com/arita37/mlmodels/blob/dev/README_model_list.md

######################################################################################

## ① Installation

Using pre-installed online Setup :

https://github.com/arita37/mlmodels/issues/101



Manual Install as editable package in Linux

```
conda create -n py36 python=3.6.5 -y
source activate py36

cd yourfolder
git clone https://github.com/arita37/mlmodels.git mlmodels
cd mlmodels
git checkout dev
```

### Check this Colab for install :
https://colab.research.google.com/drive/1sYbrXNZh9nTeizS-AuCA8RSu94B_B-RF


##### Initialize
Will copy template, dataset, example to your folder

    ml_models --init  /yourworkingFolder/


##### To test :
    ml_optim


##### To test model fitting
    ml_models
    
        
#### Actual test runs

https://github.com/arita37/mlmodels/actions

![test_fast_linux](https://github.com/arita37/mlmodels/workflows/test_fast_linux/badge.svg)

![test_fast_windows](https://github.com/arita37/mlmodels/workflows/test_fast_windows/badge.svg?branch=dev)

![ All model testing (Linux) ](https://github.com/arita37/mlmodels/workflows/code_structure_linux/badge.svg)

#######################################################################################

## Usage in Jupyter

https://github.com/arita37/mlmodels/blob/dev/README_usage.md

#######################################################################################

## CLI tools:

https://github.com/arita37/mlmodels/blob/dev/README_usage_CLI.md



####################################################################################

## Model List

https://github.com/arita37/mlmodels/blob/dev/README_model_list.md

#######################################################################################

## How to add a new model

https://github.com/arita37/mlmodels/blob/dev/README_addmodel.md

#######################################################################################

## Index of functions/methods

https://github.com/arita37/mlmodels/blob/dev/README_index_doc.txt

####################################################################################
















### LSTM example in TensorFlow ([Example notebook](mlmodels/example/1_lstm.ipynb))

#### Define model and data definitions
```python
# import library
import mlmodels


model_uri    = "model_tf.1_lstm.py"
model_pars   =  {  "num_layers": 1,
                  "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,
                }
data_pars    =  {"data_path": "/folder/myfile.csv"  , "data_type": "pandas" }
compute_pars =  { "learning_rate": 0.001, }

out_pars     =  { "path": "ztest_1lstm/", "model_path" : "ztest_1lstm/model/"}
save_pars = { "path" : "ztest_1lstm/model/" }
load_pars = { "path" : "ztest_1lstm/model/" }



#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)


#### Inference
metrics_val   =  module.fit_metrics( model, sess, data_pars, compute_pars, out_pars) # get stats
ypred         = module.predict(model, sess,  data_pars, compute_pars, out_pars)     # predict pipeline


```


---

### AutoML example in Gluon ([Example notebook](mlmodels/example/gluon_automl.ipynb))
```
# import library
import mlmodels
import autogluon as ag

#### Define model and data definitions
model_uri = "model_gluon.gluon_automl.py"
data_pars = {"train": True, "uri_type": "amazon_aws", "dt_name": "Inc"}

model_pars = {"model_type": "tabular",
              "learning_rate": ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
              "activation": ag.space.Categorical(*tuple(["relu", "softrelu", "tanh"])),
              "layers": ag.space.Categorical(
                          *tuple([[100], [1000], [200, 100], [300, 200, 100]])),
              'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),
              'num_boost_round': 10,
              'num_leaves': ag.space.Int(lower=26, upper=30, default=36)
             }

compute_pars = {
    "hp_tune": True,
    "num_epochs": 10,
    "time_limits": 120,
    "num_trials": 5,
    "search_strategy": "skopt"
}

out_pars = {
    "out_path": "dataset/"
}



#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, model_pars=model_pars, compute_pars=compute_pars, out_pars=out_pars)      


#### Inference
ypred       = module.predict(model, data_pars, compute_pars, out_pars)     # predict pipeline


```

---

### RandomForest example in Scikit-learn ([Example notebook](mlmodels/example/sklearn.ipynb))
```
# import library
import mlmodels

#### Define model and data definitions
model_uri    = "model_sklearn.sklearn.py"

model_pars   = {"model_name":  "RandomForestClassifier", "max_depth" : 4 , "random_state":0}

data_pars    = {'mode': 'test', 'path': "../mlmodels/dataset", 'data_type' : 'pandas' }

compute_pars = {'return_pred_not': False}

out_pars    = {'path' : "../ztest"}


#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
```


---

### TextCNN example in keras ([Example notebook](example/textcnn.ipynb))

```python
# import library
import mlmodels

#### Define model and data definitions
model_uri    = "model_keras.textcnn.py"

data_pars    = {"path" : "../mlmodels/dataset/text/imdb.csv", "train": 1, "maxlen":400, "max_features": 10}

model_pars   = {"maxlen":400, "max_features": 10, "embedding_dims":50}
                       
compute_pars = {"engine": "adam", "loss": "binary_crossentropy", "metrics": ["accuracy"] ,
                        "batch_size": 32, "epochs":1, 'return_pred_not':False}

out_pars     = {"path": "ztest/model_keras/textcnn/"}



#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference
data_pars['train'] = 0
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
```

---

### Using json config file for input ([Example notebook](example/1_lstm_json.ipynb), [JSON file](mlmodels/mlmodels/example/1_lstm.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_tf.1_lstm.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/example/1_lstm.json'
})

#### Load parameters and train
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model

#### Check inference
ypred       = module.predict(model, sess=sess,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline


```

---

### Using Scikit-learn's SVM for Titanic Problem from json file ([Example notebook](mlmodels/example/sklearn_titanic_svm.ipynb), [JSON file](mlmodels/example/sklearn_titanic_svm.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_sklearn.sklearn.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/example/sklearn_titanic_svm.json'
})

#### Load Parameters and Train

model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred


#### Check metrics
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)


```

---

### Using Scikit-learn's Random Forest for Titanic Problem from json file ([Example notebook](mlmodels/example/sklearn_titanic_randomForest.ipynb), [JSON file](mlmodels/example/sklearn_titanic_randomForest.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_sklearn.sklearn.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/example/sklearn_titanic_randomForest.json'
})


#### Load Parameters and Train
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference

ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred

#### Check metrics
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)

```

---

### Using Autogluon for Titanic Problem from json file ([Example notebook](mlmodels/example/gluon_automl_titanic.ipynb), [JSON file](mlmodels/example/gluon_automl.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_gluon.gluon_automl.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(
    choice='json',
    config_mode= 'test',
    data_path= '../mlmodels/example/gluon_automl.json'
)


#### Load Parameters and Train
model         =  module.Model(model_pars=model_pars, compute_pars=compute_pars)             # Create Model instance
model   =  module.fit(model, model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
model.model.fit_summary()


#### Check inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline

#### Check metrics
model.model.model_performance

import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)


```

---
---

### Using hyper-params (optuna) for Titanic Problem from json file ([Example notebook](mlmodels/example/sklearn_titanic_randomForest_example2.ipynb), [JSON file](mlmodels/example/hyper_titanic_randomForest.json))

#### Import library and functions
```python
# import library
from mlmodels.models import module_load
from mlmodels.optim import optim
from mlmodels.util import params_json_load


#### Load model and data definitions from json

###  hypermodel_pars, model_pars, ....
model_uri   = "model_sklearn.sklearn.py"
config_path = path_norm( 'example/hyper_titanic_randomForest.json'  )
config_mode = "test"  ### test/prod



#### Model Parameters
hypermodel_pars, model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)
print( hypermodel_pars, model_pars, data_pars, compute_pars, out_pars)


module            =  module_load( model_uri= model_uri )                      
model_pars_update = optim(
    model_uri       = model_uri,
    hypermodel_pars = hypermodel_pars,
    model_pars      = model_pars,
    data_pars       = data_pars,
    compute_pars    = compute_pars,
    out_pars        = out_pars
)


#### Load Parameters and Train
model         =  module.Model(model_pars=model_pars_update, data_pars=data_pars, compute_pars=compute_pars)y
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)

#### Check inference
ypred         = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred


#### Check metrics
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv( path_norm('dataset/tabular/titanic_train_preprocessed.csv') )
y = y['Survived'].values
roc_auc_score(y, ypred)


```


---

### Using LightGBM for Titanic Problem from json file ([Example notebook](mlmodels/example/model_lightgbm.ipynb), [JSON file](mlmodels/example/lightgbm_titanic.json))

#### Import library and functions
```python
# import library
import mlmodels
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm
import json

#### Load model and data definitions from json
# Model defination
model_uri    = "model_sklearn.model_lightgbm.py"
module        =  module_load( model_uri= model_uri)

# Path to JSON
data_path = '../dataset/json/lightgbm_titanic.json'  

# Model Parameters
pars = json.load(open( data_path , mode='r'))
for key, pdict in  pars.items() :
  globals()[key] = path_norm_dict( pdict   )   ###Normalize path

#### Load Parameters and Train
model = module.Model(model_pars, data_pars, compute_pars) # create model instance
model, session = module.fit(model, data_pars, compute_pars, out_pars) # fit model


#### Check inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # get predictions
ypred


#### Check metrics
metrics_val = module.fit_metrics(model, data_pars, compute_pars, out_pars)
metrics_val 

```

---




### Using Vision CNN RESNET18 for MNIST dataset  ([Example notebook](mlmodels/example/model_restnet18.ipynb), [JSON file](mlmodels/model_tch/torchhub_cnn.json))

```python
# import library
import mlmodels
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm, params_json_load
import json


#### Model URI and Config JSON
model_uri   = "model_tch.torchhub.py"
config_path = path_norm( 'model_tch/torchhub_cnn.json'  )
config_mode = "test"  ### test/prod





#### Model Parameters
hypermodel_pars, model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)
print( hypermodel_pars, model_pars, data_pars, compute_pars, out_pars)


#### Setup Model 
module         = module_load( model_uri)
model          = module.Model(model_pars, data_pars, compute_pars) 
`
#### Fit
model, session = module.fit(model, data_pars, compute_pars, out_pars)           #### fit model
metrics_val    = module.fit_metrics(model, data_pars, compute_pars, out_pars)   #### Check fit metrics
print(metrics_val)


#### Inference
ypred          = module.predict(model, session, data_pars, compute_pars, out_pars)   
print(ypred)




```
---



### Using ARMDN Time Series : Ass for MNIST dataset  ([Example notebook](mlmodels/example/model_timeseries_armdn.ipynb), [JSON file](mlmodels/model_keras/armdn.json))



```python
# import library
import mlmodels
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm, params_json_load
import json


#### Model URI and Config JSON
model_uri   = "model_keras.ardmn.py"
config_path = path_norm( 'model_keras/ardmn.json'  )
config_mode = "test"  ### test/prod




#### Model Parameters
hypermodel_pars, model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)
print( hypermodel_pars, model_pars, data_pars, compute_pars, out_pars)


#### Setup Model 
module         = module_load( model_uri)
model          = module.Model(model_pars, data_pars, compute_pars) 
`
#### Fit
model, session = module.fit(model, data_pars, compute_pars, out_pars)           #### fit model
metrics_val    = module.fit_metrics(model, data_pars, compute_pars, out_pars)   #### Check fit metrics
print(metrics_val)


#### Inference
ypred          = module.predict(model, session, data_pars, compute_pars, out_pars)   
print(ypred)



#### Save/Load
module.save(model, save_pars ={ 'path': out_pars['path'] +"/model/"})

model2 = module.load(load_pars ={ 'path': out_pars['path'] +"/model/"})



```
---



