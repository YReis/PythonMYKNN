# we are creating a kkn classifier
# we are using the iris dataset
# we are using the euclidean distance to calculate the distance between the points
import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler


def euclidianD(firstRow, secondRow):
    counter = 0
    i = 0
    while i < len(firstRow):
        counter += (firstRow[i] - secondRow[i])**2
        i += 1
    return math.sqrt(counter)


def preProcess(devs, scaler, input):
    input = input.drop(columns=["id", "name"])
    devs = devs.drop(columns=["id", "name"])
    for col in devs.columns:
        if devs[col].dtype == 'int64':
            pass
        if devs[col].dtype == 'object':
            dictRule = {}
            count = 0
            for value in devs[col].unique():
                dictRule[value] = count
                count += 1
            # print(devs[col])
            devs[col] = devs[col].replace(dictRule)
            input[col] = input[col].replace(dictRule)
            # print(dictRule)
        scaler.fit(devs[col].values.reshape(-1, 1))
        devs[col] = scaler.transform(devs[col].values.reshape(-1, 1))
        input[col] = scaler.transform(input[col].values.reshape(-1, 1))
    # print(devs[col].values.reshape(-1, 1))
    # print(devs)
    return devs, input


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
    high = {"class": "high", "count": 0}
    low = {"class": "low", "count": 0}
    medium = {"class": "medium", "count": 0}
    for i in arrClasses:
        if i == "medium":
            medium["count"] += 1
        if i == "low":
            low["count"] += 1
        else:
            high["count"] += 1
    return sorted([high, low, medium], key=lambda x: x["count"], reverse=True)[0]


def predict(devs, data, tags):
    newDevsClassified = []
    devs = devs.drop(columns="distances")
    for idx, newIncomer in data.iterrows():
        print(devs.columns)
        arrNewincomerDists = []
        for index, row in devs.iterrows():
            arrNewincomerDists.append((index, euclidianD(row, newIncomer)))
        print("arrNew", arrNewincomerDists)
        valor1 = sorted(arrNewincomerDists, key=lambda tup: tup[1])[0]
        valor2 = sorted(arrNewincomerDists, key=lambda tup: tup[1])[1]
        valor3 = sorted(arrNewincomerDists, key=lambda tup: tup[1])[2]

        classe1 = tags[valor1[0]]
        classe2 = tags[valor2[0]]
        classe3 = tags[valor3[0]]
        arrClasses = [classe1, classe2, classe3]

        # classified = highestValues(arrClasses)

        newDevsClassified.append(["NovoDev", newIncomer, classe1])
    return newDevsClassified


if __name__ == "__main__":
    devs = pd.read_json("devs.json")
    # print(devs)
    newIncomer = pd.read_json("allnewIncomers.json")
    # print(newIncomer)
    tags = devs.productivity_class
    # print(tags)
    devs = devs.drop(columns=["productivity_class"])
    scaler = MinMaxScaler()
    devs, newIncomer = preProcess(devs, scaler, newIncomer)
    # print(preProcess(devs, scaler, newIncomer))
    devs["distances"] = allDistances(devs)
    print(predict(devs, newIncomer, tags))
