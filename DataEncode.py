# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: DataEncode.py
@time: 2021-09-07 21:40
"""
import tensorflow as tf
import tensorflow.keras as keras
from official.nlp.bert import tokenization

tf.get_logger().setLevel("ERROR")
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")


def TokenizeSentence(sentence: list, tokenizer: tokenization.FullTokenizer):
    tokens = list(tokenizer.tokenize(sentence))
    tokens.append("[SEP]")
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids


def EncodeLabels(labelList: list, classNames: list, maxLength: int, padValue: str):
    labelsPadded = keras.preprocessing.sequence.pad_sequences(labelList,
                                                              maxlen=maxLength,
                                                              dtype="int32",
                                                              padding="post",
                                                              value=classNames.index(padValue))
    return labelsPadded


def BertEncode(sentences: list, tokenizer: tokenization.FullTokenizer, maxLength: int):
    numSentences = len(sentences)
    sentencesTensor = tf.ragged.constant([TokenizeSentence(s, tokenizer) for s in sentences])
    clsPrefix = [tokenizer.convert_tokens_to_ids(["[CLS]"])] * numSentences
    inputWordIds = tf.concat([clsPrefix, sentencesTensor], axis=-1)
    # check if parameter maxLength is greater than the max length of sentences
    maxSentenceLength = tf.reduce_max(inputWordIds.row_lengths()).numpy()
    print(f"Max Sentence Length: {maxSentenceLength}\n")
    assert maxLength >= maxSentenceLength
    inputMask = tf.ones_like(inputWordIds).to_tensor(shape=(None, maxLength))
    inputTypeIds = tf.zeros_like(inputWordIds).to_tensor(shape=(None, maxLength))
    encoded = dict(input_word_ids=inputWordIds.to_tensor(shape=(None, maxLength)),
                   input_mask=inputMask,
                   input_type_ids=inputTypeIds)
    return encoded


def main():
    exit(0)


if __name__ == "__main__":
    main()
