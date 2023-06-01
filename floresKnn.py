# we are creating a kkn classifier
# we are using the iris dataset
# we are using the euclidean distance to calculate the distance between the points
import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


def euclidianD(firstRow, secondRow):
    counter = 0
    i = 0
    while i < len(firstRow):
        print(firstRow[i],secondRow[i])
        counter += (firstRow[i] - secondRow[i])**2
        i += 1
    return math.sqrt(counter)


def allDistances(devs):
    arrGeneral = []
    for index, row in devs.iterrows():
        arrAllDists = []
        for index1, row1 in devs.iterrows():
            # print(index,index1)
            if index == index1:
                pass
            else:
                arrAllDists.append((index1, euclidianD(row, row1)))
        arrGeneral.append(arrAllDists)

    return arrGeneral


def highestValues(arrClasses):
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


def predict(Flores, data, tags):
    newFloresClassified = []
    for idx, newIncomer in data.iterrows():
        # print(Flores.columns)
        arrNewincomerDists = []
        for index, row in Flores.iterrows():
            arrNewincomerDists.append((index, euclidianD(row, newIncomer)))
        # print("arrNew", arrNewincomerDists)
        valor1 = sorted(arrNewincomerDists, key=lambda tup: tup[1])[0]
        valor2 = sorted(arrNewincomerDists, key=lambda tup: tup[1])[1]
        valor3 = sorted(arrNewincomerDists, key=lambda tup: tup[1])[2]

        classe1 = tags[valor1[0]]
        classe2 = tags[valor2[0]]
        classe3 = tags[valor3[0]]
        arrClasses = [classe1, classe2, classe3]

        classified = highestValues(arrClasses)

        newFloresClassified.append(classified['class'])
        print(f"Nova flor: {classified} "+"\n")
    return newFloresClassified


if __name__ == "__main__":
    flores = pd.read_csv("Iris.csv")
    tags = flores.Species
    flores = flores.drop(columns=["Id", "Species"])
    newFlores = pd.read_csv("newIris.csv")
    y_true = newFlores.Species
    newFlores = newFlores.drop(columns=["Id", "Species"])
    y_predict = predict(flores, newFlores, tags)
    print(accuracy_score(y_true,y_predict))
    # print(newFlores)
    # print(flores)
    # print(predict(devs, newIncomer, tags))
