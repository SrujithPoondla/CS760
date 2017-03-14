#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 17:20:22 2017

@author: srujithpoondla
"""

from scipy.io.arff import loadarff
import numpy
import pandas
import sys

class ArffPasesr:
    def __init__(self):
        self.data=[]
        self.attributes=[]
        self.output=[]
    
    def parse(self,filename):
        data,attributes= loadarff(filename)
        