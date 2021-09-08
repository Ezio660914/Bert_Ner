# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: Run.py
@time: 2021-09-07 23:55
"""
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import tensorflow_hub as hub
import pickle
from pathlib import Path
from official.nlp.bert import tokenization
from official.nlp.optimization import create_optimizer

"""include project files"""
from DataPreprocess import *
from DataEncode import *

tf.get_logger().setLevel("ERROR")
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")

bertDir = Path("./downloads/SavedModel/small_bert_bert_en_uncased_L-8_H-512_A-8_2")
classNames: list = GetClassNames(dataDir / rawTrainFile)


def LoadData(fileDir):
    sentenceList, labelList = PreProcessFile(fileDir)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocabDir)
    data = BertEncode(sentenceList, tokenizer)
    maxLength = data["input_word_ids"].shape[1]
    label = EncodeLabels(labelList, classNames, maxLength)
    return data, label


def BuildModel():
    # config bert model
    input1 = keras.layers.Input(shape=(None,), name="input_word_ids", dtype=tf.int32)
    input2 = keras.layers.Input(shape=(None,), name="input_mask", dtype=tf.int32)
    input3 = keras.layers.Input(shape=(None,), name="input_type_ids", dtype=tf.int32)
    bertModel = hub.KerasLayer(str(bertDir), trainable=False)
    bertInputArgs = {
        'input_word_ids': input1,
        'input_mask': input2,
        'input_type_ids': input3,
    }
    bertOutput = bertModel(bertInputArgs, training=False)
    x = bertOutput["sequence_output"]
    x_rnn = keras.layers.Bidirectional(keras.layers.LSTM(256,
                                                         return_sequences=True))(x)
    x_rnn = keras.layers.Dropout(0.2)(x_rnn)
    x = keras.layers.add([x, x_rnn])
    x = keras.layers.TimeDistributed(keras.layers.Dense(len(classNames),
                                                        activation=keras.activations.softmax))(x)
    model = keras.Model(inputs=[input1, input2, input3], outputs=x)
    return model


def main():
    trainData, trainLabel = LoadData(dataDir / rawTrainFile)
    valData, valLabel = LoadData(dataDir / rawValFile)
    testData, testLabel = LoadData(dataDir / rawTestFile)
    print("finished loading data\n")
    model = BuildModel()
    model.summary()
    # create an optimizer with learning rate schedule
    initLearningRate = 1e-5
    epochs = 5
    batchSize = 32
    trainDataSize = len(trainLabel)
    stepsPerEpoch = int(trainDataSize / batchSize)
    numTrainSteps = stepsPerEpoch * epochs
    warmupSteps = int(numTrainSteps * 0.1)
    optimizer = create_optimizer(
        init_lr=initLearningRate,
        num_train_steps=numTrainSteps,
        num_warmup_steps=warmupSteps,
        optimizer_type="adamw"
    )
    model.compile(keras.optimizers.Adam(1e-6),
                  keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])
    history = model.fit(trainData, trainLabel,
                        epochs=5,
                        validation_data=(valData, valLabel))
    model.evaluate(testData, testLabel)

    exit(0)


if __name__ == "__main__":
    main()
