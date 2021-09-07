# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: PreprocessData.py
@time: 2021-09-07 11:19
"""
from pathlib import Path
import pickle

dataDir = Path("./downloads/Resource/CoNLL")
rawTrainFile = "conllpp_train.txt"
rawValFile = "conllpp_dev.txt"
rawTestFile = "conllpp_test.txt"


def PreProcessFile(fileDir):
    with open(fileDir, "r", encoding="utf-8") as f:
        sentenceList = []
        labelList = []
        sentence = ""
        label = []
        for line in f:
            if not line == "\n":
                wordList = line.split(" ")
                sentence = sentence + wordList[0] + " "
                label.append(wordList[-1].replace("\n", ""))
            else:
                sentenceList.append(sentence.rstrip())
                labelList.append(label)
                sentence = ""
                label = []
    return sentenceList, labelList


def main():
    sentenceList, labelList = PreProcessFile(dataDir / rawTrainFile)
    # print(sentenceList[:10])
    # print(labelList)
    with open("sentenceList.pkl", "wb") as f:
        pickle.dump(sentenceList, f)
    with open("labelList.pkl", "wb") as f:
        pickle.dump(labelList, f)
    exit(0)


if __name__ == "__main__":
    main()
