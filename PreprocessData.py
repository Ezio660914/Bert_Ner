# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: PreprocessData.py
@time: 2021-09-07 11:19
"""
from pathlib import Path
import pickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

tf.get_logger().setLevel("ERROR")
dataDir = Path("./downloads/Resource/CoNLL")
rawTrainFile = "conllpp_train.txt"
rawValFile = "conllpp_dev.txt"
rawTestFile = "conllpp_test.txt"


def GetClassNames(fileDir):
    classNames = []
    with open(fileDir, "r", encoding="utf8") as f:
        for line in f:
            if not line == "\n":
                wordList = line.split(" ")
                cls = wordList[-1].replace("\n", "")
                if cls not in classNames:
                    classNames.append(cls)
    classNames.sort()
    print(classNames)
    return classNames


def GetOneHotEncoder():
    classNames = GetClassNames(dataDir / rawTrainFile)
    classNames = np.array(classNames)[:, None]
    encoder = OneHotEncoder(sparse=False, dtype=np.float32)
    encoder.fit(classNames)
    return encoder


def PreProcessFile(fileDir):
    encoder = GetOneHotEncoder()
    sentenceList = []
    labelList = []
    sentence = ""
    label = []
    classNames = []
    with open(fileDir, "r", encoding="utf-8") as f:
        for line in f:
            if not line == "\n":
                wordList = line.split(" ")
                sentence = sentence + wordList[0] + " "
                className = wordList[-1].replace("\n", "")
                label.append([className])
                if className not in classNames:
                    classNames.append(className)

            else:
                sentenceList.append(sentence.rstrip())
                oneHotLabel = encoder.transform(label).tolist()
                labelList.append(oneHotLabel)
                sentence = ""
                label = []
    labelList = tf.ragged.constant(labelList)
    return sentenceList, labelList, classNames


def main():
    sentenceList, labelList, classNames = PreProcessFile(dataDir / rawTrainFile)
    print(sentenceList[:3])
    print(labelList)
    with open("sentenceList.pkl", "wb") as f:
        pickle.dump(sentenceList, f)
    with open("labelList.pkl", "wb") as f:
        pickle.dump(labelList, f)
    with open("classNames.pkl", "wb") as f:
        pickle.dump(classNames, f)


if __name__ == "__main__":
    main()
