import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import math

class KNNClassifier:

    #object ammout for defining the neighbours to be considered

    def __init__(self, k,path):
        self.path=path
        self.k = k
    
    #data to be used

    def fit(self, Devs, tags):
        self.Devs = Devs
        self.tags = tags

    #word embedings


    def embedings(self):
        with open(self.path) as f:
            embeddings_dict = {}
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    vector = np.asarray(values[1:], dtype="float32")
                    embeddings_dict[word] = vector
                except ValueError:
                    print(f"Skipping line: {line}")
        self.embeddings_dict = embeddings_dict

        
    #euclidian dist

    def euclidianD(self, firstRow, secondRow):
        counter = 0
        i = 0
        while i < len(firstRow):
            print(firstRow[i],secondRow[i])
            counter += (firstRow[i] - secondRow[i])**2
            i += 1
        return math.sqrt(counter)
    
    #all defining the 3 closest objects

    def highestValues(self, arrClasses):
        Moderate = {"class": "Moderate", "count": 0}
        Low = {"class": "Low", "count": 0}
        High = {"class": "High", "count": 0}
        for i in arrClasses:
            if i == "High":
                High["count"] += 1
            elif i == "Low":
                Low["count"] += 1
            else:
                Moderate["count"] += 1
        return sorted([Moderate, Low, High], key=lambda x: x["count"], reverse=True)[0]
    
    #predicting the class of a object of the new data

    def _predict(self, newIncomer):
        arrNewincomerDists = [(index, self.euclidianD(newIncomer, row)) for index, row in self.Devs.iterrows()]
        k_closest = sorted(arrNewincomerDists, key=lambda tup: tup[1])[:self.k]
        arrClasses = [self.tags[i[0]] for i in k_closest]
        classified = self.highestValues(arrClasses)
        return classified['class']
    
    #prepocessing word embedings

    def preprocess(self,newIncomers):
        for index,row in newIncomers.iterrows():
            row.role = self.embeddings_dict[row.role.lower()]
            row.hobby= self.embeddings_dict[row.hobby.lower()]
            row.scholarity = self.embeddings_dict[row.scholarity.lower()]
        print(newIncomers,"testados")
        return newIncomers
        
    
    #predicting the class of each object of the new data

    def predict(self, newIncomers):
        newIncomers=self.preprocess(newIncomers)
        newIncomersClassified = [self._predict(newIncomer) for newIncomer in newIncomers.values]
        return np.array(newIncomersClassified)
        
if __name__ == "__main__":
    
    
    Devs = pd.read_json("devs.json")
    tags = Devs.productivity_class
    Devs = Devs.drop(columns=["id","name","productivity_class"])

    AllnewIncomers = pd.read_json("allnewIncomers.json")
    y_true = AllnewIncomers.productivity_class
    AllnewIncomers = AllnewIncomers.drop(columns=["id", "name"])

    classifier = KNNClassifier(k=3,path ="glove.840B.300d.txt")
    classifier.embedings()
    classifier.fit(Devs, tags)
    y_pred = classifier.predict(AllnewIncomers)

    print(accuracy_score(y_true, y_pred))
    print(y_pred)
