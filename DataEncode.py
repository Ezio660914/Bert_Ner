# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: DataEncode.py
@time: 2021-09-07 21:40
"""
import tensorflow as tf
import tensorflow.keras as keras
import pickle
from pathlib import Path
from official.nlp.bert import tokenization

tf.get_logger().setLevel("ERROR")
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")

vocabDir = Path("./downloads/SavedModel/small_bert_bert_en_uncased_L-8_H-512_A-8_2/assets/vocab.txt")
classNamesDir = Path("./classNames.pkl")
labelListDir = Path("./labelList.pkl")


def TokenizeSentence(sentence: list, tokenizer: tokenization.FullTokenizer):
    tokens = list(tokenizer.tokenize(sentence))
    tokens.append("[SEP]")
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids


def EncodeLabels():
    with open(classNamesDir, "rb") as f:
        classNames: list = pickle.load(f)
    with open(labelListDir, "rb") as f:
        labelList: list = pickle.load(f)
    labelsPadded = keras.preprocessing.sequence.pad_sequences(labelList,
                                                              dtype="float32",
                                                              padding="post",
                                                              value=classNames.index("O"))
    labelsOneHot = tf.one_hot(indices=labelsPadded,
                              depth=len(classNames),
                              axis=-1)
    return labelsOneHot


def BertEncode(sentences: list, tokenizer: tokenization.FullTokenizer):
    numSentences = len(sentences)
    sentencesTensor = tf.ragged.constant([TokenizeSentence(s, tokenizer) for s in sentences])
    clsPrefix = [tokenizer.convert_tokens_to_ids("[CLS]")] * numSentences
    inputWordIds = tf.concat([clsPrefix, sentencesTensor], axis=-1)
    inputMask = tf.ones_like(inputWordIds).to_tensor()
    inputTypeIds = tf.zeros_like(inputWordIds).to_tensor()
    encoded = dict(input_word_ids=inputWordIds.to_tensor(),
                   input_mask=inputMask,
                   input_type_ids=inputTypeIds)
    return encoded


def main():
    exit(0)


if __name__ == "__main__":
    main()
