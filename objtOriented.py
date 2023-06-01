import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import math

class KNNClassifier:

    #object ammout for defining the neighbours to be considered

    def __init__(self, k):
        self.k = k

    #data to be used

    def fit(self, Flores, tags):
        self.Flores = Flores
        self.tags = tags

    #euclidian dist

    def euclidianD(self, firstRow, secondRow):
        counter = 0
        i = 0
        while i < len(firstRow):
            counter += (firstRow[i] - secondRow[i])**2
            i += 1
        return math.sqrt(counter)
    
    #all defining the 3 closest objects

    def highestValues(self, arrClasses):
        Irissetosa = {"class": "Iris-setosa", "count": 0}
        Irisversicolor = {"class": "Iris-versicolor", "count": 0}
        Irisvirginica = {"class": "Iris-virginica", "count": 0}
        for i in arrClasses:
            if i == "Iris-virginica":
                Irisvirginica["count"] += 1
            elif i == "Iris-versicolor":
                Irisversicolor["count"] += 1
            else:
                Irissetosa["count"] += 1
        return sorted([Irissetosa, Irisversicolor, Irisvirginica], key=lambda x: x["count"], reverse=True)[0]
    
    #predicting the class of a object of the new data

    def _predict(self, newIncomer):
        arrNewincomerDists = [(index, self.euclidianD(newIncomer, row)) for index, row in self.Flores.iterrows()]
        k_closest = sorted(arrNewincomerDists, key=lambda tup: tup[1])[:self.k]
        arrClasses = [self.tags[i[0]] for i in k_closest]
        classified = self.highestValues(arrClasses)
        return classified['class']
    
    #predicting the class of each object of the new data

    def predict(self, newFlores):
        newFloresClassified = [self._predict(newIncomer) for newIncomer in newFlores.values]
        return np.array(newFloresClassified)
        
if __name__ == "__main__":
    flores = pd.read_csv("Iris.csv")
    tags = flores.Species
    flores = flores.drop(columns=["Id", "Species"])

    newFlores = pd.read_csv("newIris.csv")
    y_true = newFlores.Species
    newFlores = newFlores.drop(columns=["Id", "Species"])

    classifier = KNNClassifier(k=3)
    classifier.fit(flores, tags)
    y_pred = classifier.predict(newFlores)

    print(accuracy_score(y_true, y_pred))
    print(y_pred)
