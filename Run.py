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
import numpy as np
import pandas as pd

"""include project files"""
from DataPreprocess import *
from DataEncode import *

tf.get_logger().setLevel("ERROR")
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")
dataDir = Path("./downloads/Resource/CoNLL")
rawTrainFile = "conllpp_train.txt"
rawValFile = "conllpp_dev.txt"
rawTestFile = "conllpp_test.txt"
bertDir = Path("./downloads/SavedModel/small_bert_bert_en_uncased_L-8_H-512_A-8_2")
vocabDir = Path("./downloads/SavedModel/small_bert_bert_en_uncased_L-8_H-512_A-8_2/assets/vocab.txt")
preProcessor = PreProcessCoNLL()
classNames = preProcessor.GetLabelClasses()

checkPointDir = Path("./saved/NerModelWeights")

train = False
fineTuneBert = True
maxSeqLength = 170


class NerModel(keras.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self):
        super().__init__()
        self.base_model = self._BuildBaseModel()
        self.decoded_sequence = None
        self.potentials = None
        self.sequence_length = None
        self.chain_kernel = None

    @staticmethod
    def _BuildBaseModel():
        # config bert model
        input1 = keras.layers.Input(shape=(None,), name="input_word_ids", dtype=tf.int32)
        input2 = keras.layers.Input(shape=(None,), name="input_mask", dtype=tf.int32)
        input3 = keras.layers.Input(shape=(None,), name="input_type_ids", dtype=tf.int32)
        bertModel = hub.KerasLayer(str(bertDir), trainable=fineTuneBert, name="bert")
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
        x = keras.layers.TimeDistributed(keras.layers.Dense(len(classNames)))(x)
        crfOutput = tfa.layers.crf.CRF(len(classNames))(x)
        baseModel = keras.Model(inputs=[input1, input2, input3], outputs=crfOutput)
        return baseModel

    def get_config(self):
        return self.base_model.get_config()

    def call(self, inputs, training=None, mask=None):
        self.decoded_sequence, self.potentials, self.sequence_length, self.chain_kernel = self.base_model(inputs,
                                                                                                          training,
                                                                                                          mask)
        return self.decoded_sequence

    def summary(self, line_length=None, positions=None, print_fn=None):
        return self.base_model.summary(line_length=None, positions=None, print_fn=None)

    @staticmethod
    def unpack_data(data):
        if len(data) == 2:
            return data[0], data[1], None
        elif len(data) == 3:
            return data
        else:
            raise TypeError("Expected data to be a tuple of size 2 or 3.")

    def compute_loss(self, x, y, sample_weight, training=False):
        # call forward
        self(x, training=training)
        # we now add the CRF loss:
        crf_loss = -tfa.text.crf_log_likelihood(self.potentials,
                                                y,
                                                self.sequence_length,
                                                self.chain_kernel)[0]

        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight

        # compute accuracy
        equalMatrix = tf.equal(y, self.decoded_sequence)
        equalMatrix = tf.cast(equalMatrix, dtype=tf.float32)
        accuracy = tf.reduce_mean(equalMatrix)
        return tf.reduce_mean(crf_loss), accuracy

    def train_step(self, data):
        x, y, sample_weight = self.unpack_data(data)

        with tf.GradientTape() as tape:
            crf_loss, accuracy = self.compute_loss(
                x, y, sample_weight, training=True
            )
            total_loss = crf_loss + sum(self.losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"crf_loss": crf_loss, "accuracy": accuracy}

    def test_step(self, data):
        x, y, sample_weight = self.unpack_data(data)
        crf_loss, accuracy = self.compute_loss(x, y, sample_weight)
        return {"crf_loss": crf_loss, "accuracy": accuracy}


def LoadData(fileDir):
    sentenceList, labelList = preProcessor.PreProcessFile(fileDir)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocabDir)
    data = BertEncode(sentenceList, tokenizer, maxSeqLength)
    label = EncodeLabels(labelList, classNames, maxSeqLength)
    return data, label


def MakePrediction(model: keras.Model, sentenceList: list):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocabDir)
    sentencesLength = tf.ragged.constant([TokenizeSentence(s, tokenizer) for s in sentenceList]).row_lengths()
    # exclude the last [SEP] token
    sentencesLength = sentencesLength - 1
    data = BertEncode(sentenceList, tokenizer, maxSeqLength)
    classNamesArray = np.array(classNames)
    labelIdPred = model.predict(data)
    labelPred = classNamesArray[labelIdPred]
    labelPred = tf.RaggedTensor.from_tensor(tensor=labelPred,
                                            lengths=sentencesLength)
    labelPred = labelPred.to_list()
    return labelPred


def main():
    trainData, trainLabel = LoadData(preProcessor.dataDir / preProcessor.rawTrainFile)
    valData, valLabel = LoadData(preProcessor.dataDir / preProcessor.rawValFile)
    testData, testLabel = LoadData(preProcessor.dataDir / preProcessor.rawTestFile)
    print("finished loading data\n")
    print(len(trainLabel), len(valLabel), len(testLabel))
    tf.random.set_seed(2021)
    model = NerModel()
    model.summary()
    # create an optimizer with learning rate schedule
    initLearningRate = 1e-5
    epochs = 2
    batchSize = 16
    trainDataSize = len(trainLabel)
    stepsPerEpoch = int(trainDataSize / batchSize)
    numTrainSteps = stepsPerEpoch * epochs
    warmupSteps = int(numTrainSteps * 0.1)
    optimizer = create_optimizer(init_lr=initLearningRate,
                                 num_train_steps=numTrainSteps,
                                 num_warmup_steps=warmupSteps,
                                 optimizer_type="adamw")
    model.compile(optimizer)
    ckptCallback = keras.callbacks.ModelCheckpoint(filepath=str(checkPointDir),
                                                   monitor="val_crf_loss",
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=True)
    if train:
        history = model.fit(trainData,
                            trainLabel,
                            batch_size=batchSize,
                            epochs=epochs,
                            validation_data=(valData, valLabel),
                            callbacks=[ckptCallback])
    model.load_weights(str(checkPointDir))
    model.evaluate(testData, testLabel)
    testText = [
        "Mr. Egeland said the latest figures show 1.8 million people are in need of food assistance - with the need greatest in Indonesia , Sri Lanka , the Maldives and India .",
        "Prime Minister Geir Haarde has refused to resign or call for early elections .",
        "The British man blames Iceland 's economic calamity on commercial bankers .",

    ]
    labelPred = MakePrediction(model, testText)

    print(labelPred)
    exit(0)


if __name__ == "__main__":
    main()
