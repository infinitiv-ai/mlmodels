  
import re
import spacy
import torch
import torchtext
from torchtext.data import Field, Pipeline
from mlmodels.util import log
from time import sleep


def test_pandas_fillna(data, **args):
    return data.fillna(**args)


def test_onehot_sentences(data, max_len):
    return (
        lambda df, max_len: (
            lambda d, ml, word_dict, sentence_groups: np.array(
                keras.preprocessing.sequence.pad_sequences(
                    [
                        [word_dict[x] for x in sw]
                        for sw in [y.values for _, y in sentence_groups["Word"]]
                    ],
                    ml,
                    padding="post",
                    value=0,
                    dtype="int",
                ),
                dtype="O",
            )
        )(
            data,
            max_len,
            {y: x for x, y in enumerate(["PAD", "UNK"] + list(data["Word"].unique()))},
            data.groupby("Sentence #"),
        )
    )(data, max_len)


def test_word_count(data):
    return data["Word"].nunique() + 2


def test_word_categorical_labels_per_sentence(data, max_len):
    return (
        lambda df, max_len: (
            lambda d, ml, c, tag_dict, sentence_groups: np.array(
                [
                    keras.utils.to_categorical(i, num_classes=c + 1)
                    for i in keras.preprocessing.sequence.pad_sequences(
                        [
                            [tag_dict[w] for w in s]
                            for s in [y.values for _, y in sentence_groups["Tag"]]
                        ],
                        ml,
                        padding="post",
                        value=0,
                    )
                ]
            )
        )(
            data,
            max_len,
            data["Tag"].nunique(),
            {y: x for x, y in enumerate(["PAD"] + list(data["Tag"].unique()))},
            data.groupby("Sentence #"),
        )
    )(data, max_len)


# textcnn_dataloader
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def imdb_spacy_tokenizer(text, lang="en"):
    disable = (
        "tagger",
        "parser",
        "ner",
        "textcat" "entity_ruler",
        "sentencizer",
        "merge_noun_chunks",
        "merge_entities",
        "merge_subtokens",
    )

    if "spacy_cache" not in globals():
        global spacy_cache
        spacy_cache = {}

    try:
        spacy_en = spacy_cache[((f"{lang}_core_web_sm"), disable)]
    except:
        try:
            spacy_en = spacy.load(f"{lang}_core_web_sm", disable)
        except:
            log(f"Download {lang}")
            os.system(f"python -m spacy download {lang}")
            sleep(5)
            spacy_en = spacy.load(f"{lang}_core_web_sm", disable=disable)
        spacy_cache[((f"{lang}_core_web_sm"), disable)] = spacy_en
    return [tok.text for tok in spacy_en.tokenizer(text)]
