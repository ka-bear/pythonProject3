import json
import sys
import time
import warnings

import sklearn
from sklearn import mixture
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
import sklearn.neighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import datasets, mixture
import sklearn.cluster as cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import requests
from flask import Flask, request
import pandas as pd


def classify(data):
    global x

    dataBroken = [s for s in data.split(" ")]
    first = [s for s in dataBroken[0].split(",")[1:]]
    Data = [s for s in dataBroken[1:]]

    ActData = []
    TestData = []

    random.shuffle(Data)

    num = int(float(first[0]) * len(Data))
    xmax = 0
    ymax = 0
    xmin = 0
    ymin = 0

    for j in Data[:num]:
        lst = [float(k) for k in str(j).split(",")]
        lst[2] = int(lst[2])
        if lst[0] > xmax:
            xmax = lst[0]
        if lst[1] > ymax:
            ymax = lst[1]
        if lst[0] < xmin:
            xmin = lst[0]
        if lst[1] < ymin:
            ymin = lst[1]
        ActData.append(lst)
    for j in Data[num:]:
        lst = [float(k) for k in str(j).split(",")]
        lst[2] = int(lst[2])
        if lst[0] > xmax:
            xmax = lst[0]
        if lst[1] > ymax:
            ymax = lst[1]
        if lst[0] < xmin:
            xmin = lst[0]
        if lst[1] < ymin:
            ymin = lst[1]
        TestData.append(lst)

    df = pd.DataFrame(ActData, columns=["x", "y", "option"])
    X = df.loc[:, ["x", "y"]]
    Y = df.option

    tdf = pd.DataFrame(TestData, columns=["x", "y", "option"])
    XX = tdf.loc[:, ["x", "y"]]
    YY = tdf.option

    '''i = 1
    # iterate over datasets
        # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)



    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]


        i += 1
    '''
    x = ""
    if first[1] == "0":
        clf = KNeighborsClassifier(n_neighbors=int(first[2]), leaf_size=int(first[3]), p=int(first[4]),weights= first[5])
        clf.fit(X, Y)
        x += str(clf.score(XX, YY))
        x += "\\n"
        x += "\\n"
        for i in range(int(ymin - 1), int(ymax + 1)):
            for j in range(int(xmin - 1), int(xmax + 1)):
                x += str(clf.predict([[i, j]]))[1:-1] + ","
            x += "\\n"

    elif first[1] == "1":
        clf = SVC(C=float(first[2]),kernel= first[3], degree=int(first[4]),coef0= float(first[5]))
        clf.fit(X, Y)
        x += str(clf.score(XX, YY))
        x += "\n"
        x += "\n"
        for i in range(int(ymin - 1), int(ymax + 1)):
            for j in range(int(xmin - 1), int(xmax + 1)):
                x += str(clf.predict([[i, j]]))[1:-1] + ","
            x += "\\n"

    elif first[1] == "2":
        hidden = [int(j) for j in first[3:]]
        hidden = tuple(hidden)
        clf = MLPClassifier(hidden_layer_sizes=hidden, solver=first[2])
        clf.fit(X, Y)
        x += str(clf.score(XX, YY))
        x += "\n"
        x += "\n"
        for i in range(int(ymin - 1), int(ymax + 1)):
            for j in range(int(xmin - 1), int(xmax + 1)):
                x += str(clf.predict([[i, j]]))[1:-1] + ","
            x += "\\n "
    return x


def Cluster(data):
    global x

    dataBroken = [s for s in data.split(" ")]
    first = [s for s in dataBroken[0].split(",")]
    Data = [s for s in dataBroken[1:]]

    ActData = []

    for j in Data:
        lst = [float(k) for k in str(j).split(",")]
        ActData.append(lst)

    df = pd.DataFrame(ActData, columns=["x", "y"])
    X = df.loc[:, ["x", "y"]]

    kneighbors_graph(df, n_neighbors=1, include_self=False)
    x = ""
    if first[1] == "0":
        kMeans = cluster.KMeans(n_clusters=int(first[2]), init=first[3], max_iter=int(first[4]))
        result = kMeans.fit(X)
        x += str(result.labels_)
    elif first[1] == "1":
        kMeans = cluster.Birch(n_clusters=int(first[2]), threshold=float(first[3]), branching_factor=int(first[4]))
        result = kMeans.fit(X)
        x += str(result.labels_)
    elif first[1] == "2":
        kMeans = cluster.AgglomerativeClustering(n_clusters=int(first[2]),linkage=first[3],affinity=first[4])
        result = kMeans.fit(X)
        x += str(result.labels_)
    return x


def regress(data):
    global x

    dataBroken = [s for s in data.split(" ")]
    first = [s for s in dataBroken[0].split(",")]
    Data = [s for s in dataBroken[1:]]

    ActData = []

    xmax = 0
    xmin = 0

    print(dataBroken)
    print(first)
    print(Data)
    random.shuffle(Data)

    for j in Data:
        lst = [float(k) for k in str(j).split(",")]
        if lst[0] > xmax:
            xmax = lst[0]
        if lst[0] < xmin:
            xmin = lst[0]
        ActData.append(lst)

    df = pd.DataFrame(ActData, columns=["x", "y"])
    X = df.loc[:, ["x"]]
    Y = df.loc[:, ["y"]]

    x = ""
    if first[1] == "0":
        clf = SVR(kernel=first[2], degree=int(first[3]), coef0=float(first[4]), C=float(first[5]),
                  max_iter=int(first[6]))
        clf.fit(X, Y)

        lst = []
        for j in range(int(xmin - 1), int(xmax + 1)):
            lst.append(j)
        tdf = pd.DataFrame(lst, columns=["x"])
        x += clf.predict(tdf)
        x += "\\n"

    elif first[1] == "1":
        hidden = [int(j) for j in first[3:]]
        hidden = tuple(hidden)
        clf = MLPRegressor(hidden_layer_sizes=hidden, solver=first[2])
        clf.fit(X, Y)

        lst = []
        for j in range(int(xmin - 1), int(xmax + 1)):
            lst.append(j)

        tdf = pd.DataFrame(lst, columns=["x"])
        x += str(clf.predict(tdf))
        print(xmin, xmax, lst, tdf, clf.predict(tdf))
        x += "\\n"
    print()
    print(x)
    return x


x = "bazinga"

app = Flask(__name__)


@app.route('/')
def hello():
    global x
    return x


@app.route('/', methods=['POST'])
def example():
    print("hiih")
    """if request.json == "exit":
        sys.exit()"""
    global x
    x = str(request.data)[2:len(str(request.data)) - 1]
    print(x[0])
    if x[0] == "0":
        classify(x)
    elif x[0] == "1":
        Cluster(x)
    elif x[0] == "2":
        regress(x)

    return x


if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=9002)
