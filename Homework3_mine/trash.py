import numpy as np
import Parse_file


class Trashnet(object):
    def __init__(self, inputSize, hiddenSize, learningrate):
        self.inputsize = inputSize
        self.hiddensize = hiddenSize
        self.learningrate = learningrate

        # alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

        # hiddenSize = 32


        # compute sigmoid nonlinearity

    def sigmoid(self, x):
        output = 1 / (1 + np.exp(-x))
        return output


        # convert output of sigmoid function to its derivative

    def sigmoid_output_to_derivative(self, output):
        return output * (1 - output)

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    def feedforward(self, X, y):
        # randomly initialize our weights with mean 0
        synapse_0 = 2 * np.random.random((3, self.hiddenSize)) - 1
        synapse_1 = 2 * np.random.random((self.hiddenSize, 1)) - 1

        for j in xrange(60000):

            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0, synapse_0))
            layer_2 = self.sigmoid(np.dot(layer_1, synapse_1))

            # how much did we miss the target value?
            layer_2_error = layer_2 - y

            if (j % 10000) == 0:
                print "Error after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error)))

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * self.sigmoid_output_to_derivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * self.sigmoid_output_to_derivative(layer_1)

            synapse_1 -= self.learningrate * (layer_1.T.dot(layer_2_delta))
            synapse_0 -= self.learningrate * (layer_0.T.dot(layer_1_delta))
