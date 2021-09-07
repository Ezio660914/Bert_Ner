# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: DataPreprocess.py
@time: 2021-09-07 11:19
"""
from pathlib import Path
import pickle

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
    return classNames


def PreProcessFile(fileDir):
    sentenceList = []
    labelList = []
    sentence = ""
    label = []
    classNames = GetClassNames(fileDir)
    with open(fileDir, "r", encoding="utf-8") as f:
        for line in f:
            if not line == "\n":
                wordList = line.split(" ")
                sentence = sentence + wordList[0] + " "
                className = wordList[-1].replace("\n", "")
                label.append(classNames.index(className))
            else:
                sentenceList.append(sentence.rstrip())
                labelList.append(label)
                sentence = ""
                label = []
    return sentenceList, labelList


def main():
    fileDir = dataDir / rawTrainFile
    sentenceList, labelList = PreProcessFile(fileDir)
    classNames = GetClassNames(fileDir)
    print(classNames)
    print(sentenceList[:3])
    print(labelList[:3])
    with open("sentenceList.pkl", "wb") as f:
        pickle.dump(sentenceList, f)
    with open("labelList.pkl", "wb") as f:
        pickle.dump(labelList, f)
    with open("classNames.pkl", "wb") as f:
        pickle.dump(classNames, f)


if __name__ == "__main__":
    main()
