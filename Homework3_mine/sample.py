# import CrossValidation
# import arffparser
import Parse_file
import stratifiednfold
import NeuralNetwork
import numpy as np
import trash as tr

train_parser = Parse_file.ARFF_Parse()
train_data = train_parser.parse('sonar.arff')

# create_train = stratifiednfold.create_stratified_folds(10)
# folds=create_train.create_stratified_data(train_data[0],train_data[1],train_data[2])


all_instance_features = train_data[0]['Class']
train_data[0] = train_data[0].drop('Class', 1)


# bias=[]
# for i in range(train_data[0].shape[0]):
#     bias.append(1)
# (train_data[0])['bias']=bias
# print train_data


train_data_list = train_data[0].as_matrix()
# train_data_list= np.hstack((train_data_list, np.atleast_2d(bias).T))


# train_data_list=train_data_list.append(bias)
#

nn = NeuralNetwork.NeuralNet(len(train_data_list[1]), len(train_data_list[1]), 1, 0.001)
y = nn.actual_output(all_instance_features, train_data[2])

for i in range(60000):
    # for j in range(folds):
            error = nn.feedforward(train_data_list,y)
            # print layers[-1]
            if (i % 1000) == 0:
                print "Error after " + str(i) + " iterations:" + str(np.mean(np.abs(error)))




# alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# hiddenSize = 60
#
#
# # compute sigmoid nonlinearity
# def sigmoid(x):
#     output = 1 / (1 + np.exp(-x))
#     return output
#
#
# # convert output of sigmoid function to its derivative
# def sigmoid_output_to_derivative(output):
#     return output * (1 - output)
#
#
# X = train_data_list
# print X
# # X = np.array([[0, 0, 1],
# #               [0, 1, 1],
# #               [1, 0, 1],
# #               [1, 1, 1]])
# #
# # y = np.array([[0],
# #               [1],
# #               [1],
# #               [0]])
#
#
# # randomly initialize our weights with mean 0
# synapse_0 = 2 * np.random.random((61, hiddenSize)) - 1
# synapse_1 = 2 * np.random.random((hiddenSize+1, 1)) - 1
#
# for j in xrange(60000):
#     alpha = alphas[0]
#
#     # Feed forward through layers 0, 1, and 2
#     layer_0 = X
#     layer_1 = sigmoid(np.dot(layer_0, synapse_0))
#     layer_1=np.hstack((layer_1, np.atleast_2d(bias).T))
#     layer_2 = sigmoid(np.dot(layer_1, synapse_1))
#
#     # how much did we miss the target value?
#     layer_2_error = layer_2 - y
#
#     if (j % 1000) == 0:
#         print "Error after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error)))
#
#     # in what direction is the target value?
#     # were we really sure? if so, don't change too much.
#     layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
#
#     # how much did each l1 value contribute to the l2 error (according to the weights)?
#     layer_1_error = layer_2_delta.dot(synapse_1.T)
#
#     # in what direction is the target l1?
#     # were we really sure? if so, don't change too much.
#     layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
#
#     synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
#     synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))
