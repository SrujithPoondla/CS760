import parse_arff
import pandas as pd
import NeuralNetwork

class StratifiedCrossValidation:
    def __init__(self,numberoffolds):
        self.numberoffolds = numberoffolds
        self.stratified_data=[numberoffolds]
        
        
    def createstratifiedfolds(self,data,attributes):
        self.numberofinstances = data.size
        
        
        