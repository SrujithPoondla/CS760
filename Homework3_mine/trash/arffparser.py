#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 17:20:22 2017

@author: srujithpoondla
"""
from scipy.io.arff import loadarff
import numpy as np
import pandas as pd

class ArffPasesr:
    def __init__(self):
        self.data=[]
        self.attributes=[]
        self.output=[]
    
    def parse(self,filename):
        x=[]
        y=[]
        gen=[]
        meta,train= loadarff(filename)
        out = train.__getitem__('Class')
        train_data = meta[train.names()[:-1]]
        print train_data.view(np.float).reshape(meta.shape + (-1,))
        print len(out[1])
        print len(train.names())
        print len(meta)
        print meta[train.names()[-1]]
        print list(meta)[0]
        classes= train.__getitem__("Class")
        print classes[1][0]