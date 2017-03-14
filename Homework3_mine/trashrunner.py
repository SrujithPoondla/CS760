import trash
import numpy as np
import NeuralNetwork
import stratifiednfold


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

nnet = trash.Trashnet(3,3,3)
n =stratifiednfold.