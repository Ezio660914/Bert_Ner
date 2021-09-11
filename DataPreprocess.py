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
import xml.etree.ElementTree as ET
import conllu


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


class PreProcessGSDSimp(PreProcessBase):
    def __init__(self):
        self.dataDir = Path("./downloads/Resource/UD_Chinese-GSDSimp-master")
        self.rawTrainFile = "zh_gsdsimp-ud-train.conllu"
        self.rawValFile = "zh_gsdsimp-ud-dev.conllu"
        self.rawTestFile = "zh_gsdsimp-ud-test.conllu"
        self.includedClass = ["ADJ", "NOUN", "VERB"]

    def GetLabelClasses(self):
        statsDir = "downloads/Resource/UD_Chinese-GSDSimp-master/stats.xml"
        tree = ET.ElementTree(file=statsDir)
        classesNames = [child.attrib["name"] for child in list(list(tree.getroot())[4])]
        # BIO encode class names
        classesNames_BIO = set()
        for cls in classesNames:
            if cls not in self.includedClass:
                classesNames_BIO.add("O")
            else:
                classesNames_BIO.add("B-" + cls)
                classesNames_BIO.add("I-" + cls)
        classesNames_BIO = list(classesNames_BIO)
        classesNames_BIO.sort()
        return classesNames_BIO

    def PreProcessFile(self, fileDir):
        classesNames = self.GetLabelClasses()
        sentenceList = []
        labelList = []
        with open(fileDir, "r", encoding="utf8") as file:
            sentences = conllu.parse_incr(file)
            for s in sentences:
                sentenceStr = s.metadata["text"]
                sentenceList.append(sentenceStr)
                label = []
                for token in s:
                    isFirst = True
                    for i in range(len(token["form"])):
                        if token["upos"] not in self.includedClass:
                            label.append(classesNames.index("O"))
                            continue
                        if isFirst:
                            label.append(classesNames.index("B-" + token["upos"]))
                            isFirst = False
                        else:
                            label.append(classesNames.index("I-" + token["upos"]))
                        if token["upos"] == "NUM":
                            break
                labelList.append(label)
        return sentenceList, labelList


def main():
    dataDir = Path("./downloads/Resource/UD_Chinese-GSDSimp-master")
    rawTrainFile = "zh_gsdsimp-ud-train.conllu"
    rawValFile = "zh_gsdsimp-ud-dev.conllu"
    rawTestFile = "zh_gsdsimp-ud-test.conllu"

    classNamesDir = Path("./classNames.pkl")
    labelListDir = Path("./labelList.pkl")
    sentenceListDir = Path("./sentenceList.pkl")

    preProcessor = PreProcessGSDSimp()

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
