import Parse_file
import random
import numpy as np

class create_stratified_folds:
    def __init__(self,num_folds):
        self.num_folds=num_folds
        self.stratified_data=[]
        self.num_class1=0
        self.num_class2=0
        self.class1=[]
        self.class2=[]
        self.temp_list=[]


    def create_stratified_data(self,data,attributes,classes):


        class_label = data['Class']
        # data = data[attributes[:-1]]
        # print data
        for i in range(len(class_label)):
            if class_label[i]==classes[0]:
                self.num_class1=self.num_class1+1
                self.class1.append(data.ix[i])
            elif class_label[i]==classes[1]:
                self.num_class2=self.num_class2+1
                self.class2.append(data.ix[i])

        randomized_class1 = random.sample(self.class1,len(self.class1))
        randomized_class2 = random.sample(self.class2,len(self.class2))

        num_instances_fold = len(data)/self.num_folds
        for i in range(self.num_folds):
            for j in range(num_instances_fold):
                self.temp_list.append(randomized_class1[j])
                self.temp_list.append(randomized_class2[j])
            self.stratified_data.append(self.temp_list)
            self.temp_list=[]

        myarray = np.asarray(self.stratified_data)
        print myarray