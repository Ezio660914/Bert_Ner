# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: DataPreprocess.py
@time: 2021-09-07 11:19
"""
"""
This file is used for preprocessing the CoNLL dataset 
"""
from pathlib import Path
import pickle


class PreProcessBase:
    def GetLabelClasses(self):
        raise NotImplementedError

    def PreProcessFile(self, fileDir):
        raise NotImplementedError


class PreProcessCoNLL(PreProcessBase):
    def __init__(self):
        self.dataDir = Path("./downloads/Resource/CoNLL")
        self.rawTrainFile = "conllpp_train.txt"
        self.rawValFile = "conllpp_dev.txt"
        self.rawTestFile = "conllpp_test.txt"

    def GetLabelClasses(self):
        fileDir = self.dataDir / self.rawTrainFile
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

    def PreProcessFile(self, fileDir):
        sentenceList = []
        labelList = []
        sentence = ""
        label = []
        classNames = self.GetLabelClasses()
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
    dataDir = Path("./downloads/Resource/CoNLL")
    rawTrainFile = "conllpp_train.txt"
    rawValFile = "conllpp_dev.txt"
    rawTestFile = "conllpp_test.txt"

    classNamesDir = Path("./classNames.pkl")
    labelListDir = Path("./labelList.pkl")
    sentenceListDir = Path("./sentenceList.pkl")

    preProcessor = PreProcessCoNLL()

    fileDir = dataDir / rawTrainFile
    sentenceList, labelList = preProcessor.PreProcessFile(fileDir)
    classNames = preProcessor.GetLabelClasses()
    print(classNames)
    print(sentenceList[:3])
    print(labelList[:3])
    with open(sentenceListDir, "wb") as f:
        pickle.dump(sentenceList, f)
    with open(labelListDir, "wb") as f:
        pickle.dump(labelList, f)
    with open(classNamesDir, "wb") as f:
        pickle.dump(classNames, f)


if __name__ == "__main__":
    main()
