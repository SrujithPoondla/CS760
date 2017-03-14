#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:11:17 2017

@author: srujithpoondla
"""
import numpy as np


class NeuralNet(object):
    def __init__(self, inputSize, hiddenSize, outputSize, lrate):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.learningRate = lrate
        self.expected_output = []

    def sigmoid(self, val):
        expo = np.exp(-val)
        return 1 / (1 + expo)

    def sigmoidderiv(self, val):
        return val * (1 - val)

    def actual_output(self, actualoutput, classes):
        for item in range(len(actualoutput)):
            if actualoutput[item] == classes[0]:
                self.expected_output.append([0])
            elif actualoutput[item] == classes[1]:
                self.expected_output.append([1])
        self.expected_output = np.array(self.expected_output)
        return self.expected_output

    def feedforward(self, val, actual_output):
        input_weights = 2 * np.random.random((self.inputSize, self.hiddenSize)) - 1
        output_weights = 2 * np.random.random((self.hiddenSize, 1)) - 1

        # Propogate inputs through network
        input_layer = val
        hidden_layer = self.sigmoid(np.dot(input_layer, input_weights))
        output_layer = self.sigmoid(np.dot(hidden_layer, output_weights))

        output_layer_error = output_layer - actual_output
        output_delta = output_layer_error * self.sigmoidderiv(output_layer)

        input_layer_error = output_delta.dot(output_weights.T)

        input_delta = input_layer_error * self.sigmoidderiv(hidden_layer)

        output_weights = output_weights - self.learningRate * (hidden_layer.T.dot(output_delta))
        input_weights = input_weights - self.learningRate * (input_layer.T.dot(input_delta))

        return output_layer_error
