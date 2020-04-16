import keras
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


class Preprocess_namentity:
    def __init__(self,max_len,**args):
        self.max_len = max_len
    
    def compute(self,df):
        df = df.fillna(method='ffill')
        ##### Get sentences
        agg = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                    s['POS'].values.tolist(),
                                                    s['Tag'].values.tolist())]
        grouped = df.groupby("Sentence #").apply(agg)
        sentences = [s for s in grouped]
    
        # Getting unique words and labels from data
        words = list(df['Word'].unique())
        tags = list(df['Tag'].unique())
        # Dictionary word:index pair
        # word is key and its value is corresponding index
        word_to_index = {w: i + 2 for i, w in enumerate(words)}
        word_to_index["UNK"] = 1
        word_to_index["PAD"] = 0
    
        # Dictionary lable:index pair
        # label is key and value is index.
        tag_to_index = {t: i + 1 for i, t in enumerate(tags)}
        tag_to_index["PAD"] = 0
    
        idx2word = {i: w for w, i in word_to_index.items()}
        idx2tag = {i: w for w, i in tag_to_index.items()}
    
    
        # Converting each sentence into list of index from list of tokens
        X = [[word_to_index[w[0]] for w in s] for s in sentences]
    
        # Padding each sequence to have same length  of each word
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=word_to_index["PAD"])
    
        # Convert label to index
        y = [[tag_to_index[w[2]] for w in s] for s in sentences]
    
        # padding
        y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=tag_to_index["PAD"])
        num_tag = df['Tag'].nunique()
        # One hot encoded labels
        y = np.array([to_categorical(i, num_classes=num_tag + 1) for i in y])
        self.data = {"X":X,"y":y,"word_count":len(df['Word'].unique())+2}
    def get_data(self):
        return self.data

