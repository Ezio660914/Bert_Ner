# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: Run.py
@time: 2021-09-07 23:55
"""
from abc import ABC

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
        bertModel = hub.KerasLayer(str(bertDir), trainable=False, name="Bert")
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
        return {"val_crf_loss": crf_loss, "val_accuracy": accuracy}


def LoadData(fileDir):
    sentenceList, labelList = PreProcessFile(fileDir)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocabDir)
    data = BertEncode(sentenceList, tokenizer)
    maxLength = data["input_word_ids"].shape[1]
    label = EncodeLabels(labelList, classNames, maxLength)
    return data, label


def main():
    trainData, trainLabel = LoadData(dataDir / rawTrainFile)
    valData, valLabel = LoadData(dataDir / rawValFile)
    testData, testLabel = LoadData(dataDir / rawTestFile)
    print("finished loading data\n")
    model = NerModel()
    model.summary()
    # create an optimizer with learning rate schedule
    initLearningRate = 1e-5
    epochs = 2
    batchSize = 32
    trainDataSize = len(trainLabel)
    stepsPerEpoch = int(trainDataSize / batchSize)
    numTrainSteps = stepsPerEpoch * epochs
    warmupSteps = int(numTrainSteps * 0.1)
    optimizer = create_optimizer(
        init_lr=initLearningRate,
        num_train_steps=numTrainSteps,
        num_warmup_steps=warmupSteps,
        optimizer_type="adamw")
    model.compile(optimizer)
    history = model.fit(trainData,
                        trainLabel,
                        batch_size=batchSize,
                        epochs=epochs,
                        validation_data=(valData, valLabel))
    model.evaluate(testData, testLabel)

    exit(0)


if __name__ == "__main__":
    main()
