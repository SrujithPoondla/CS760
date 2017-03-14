from scipy.io.arff import loadarff
import stratifiednfold
import pandas as pd

class ARFF_Parse:
    def __init__(self):
        self.data = []
        self.attributes=[]
        self.classes=[]
        self.returnlist=[]

    def parse(self,filename):
        self.data,meta = loadarff(filename)
        self.data=pd.DataFrame(self.data)
        self.returnlist.append(self.data)
        self.attributes = meta.names()
        self.returnlist.append(self.attributes)
        self.classes= meta.__getitem__("Class")[1]
        self.returnlist.append(self.classes)
        return self.returnlist






