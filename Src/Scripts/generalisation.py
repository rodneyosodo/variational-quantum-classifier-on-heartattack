#!/usr/bin/env python
# coding: utf-8
from qiskit.aqua.components.optimizers import COBYLA, ADAM, SPSA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, ZFeatureMap, PauliFeatureMap

from Benchmarking import Benchmark, normalize_data
import csv


import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import warnings

warnings.filterwarnings("ignore")

best_models = [
    {"Featuremap": "ZFeatureMap(4, reps=2)", "Opt": "SPSA(max_trials=50)", "vdepth": 5},
    {"Featuremap": "ZFeatureMap(4, reps=2)", "Opt": "SPSA(max_trials=50)", "vdepth": 3},
    {"Featuremap": "ZFeatureMap(4, reps=2)", "Opt": "COBYLA(maxiter=50)", "vdepth": 3},
    {"Featuremap": "ZFeatureMap(4, reps=2)", "Opt": "SPSA(max_trials=50)", "vdepth": 1},
    {"Featuremap": "ZFeatureMap(4, reps=1)", "Opt": "COBYLA(maxiter=50)", "vdepth": 1},
    {"Featuremap": "ZZFeatureMap(4, reps=1)", "Opt": "SPSA(max_trials=50)", "vdepth": 5},
    {"Featuremap": "ZFeatureMap(4, reps=2)", "Opt": "COBYLA(maxiter=50)", "vdepth": 5},
    {"Featuremap": "ZFeatureMap(4, reps=1)", "Opt": "SPSA(max_trials=50)", "vdepth": 3},
    {"Featuremap": "ZFeatureMap(4, reps=1)", "Opt": "SPSA(max_trials=50)", "vdepth": 5},
    {"Featuremap": "ZFeatureMap(4, reps=1)", "Opt": "COBYLA(maxiter=50)", "vdepth": 3},
]

def normalize_data(dataPath = "../../Data/Processed/iris_csv.csv"):
    """
    Normalizes the data
    """
    # Reads the data
    data = pd.read_csv(dataPath)
    data = shuffle(data, random_state=42)
    if dataPath.__contains__("iris"):
        X, Y = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values, data['target'].values
    elif dataPath.__contains__("wine"):
        X, Y = data[['alcohol', 'flavanoids', 'color_intensity', 'proline', 'target']].values, data['target'].values
    # normalize the data
    scaler = MinMaxScaler(feature_range=(-2 * np.pi, 2 * np.pi))
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test

def main():
    data = {}
    dataPaths = ["../../Data/Processed/iris_csv.csv", "../../Data/Processed/winedata.csv"]
    for dataPath in dataPaths:
        x_train, x_test, y_train, y_test = normalize_data(dataPath = dataPath)
        for model in best_models:
            print("FE: {}\tDepth: {}\tOpt: {}".format(model['Featuremap'], model['vdepth'], model['Opt']))
            test_benchmark = Benchmark(optimizer=eval(model['Opt']), variational_depth=model['vdepth'], feature_map=eval(model['Featuremap']), X_train=x_train, X_test=x_test, Y_train=y_train, Y_test=y_test)
            test_benchmark.run()
            data_list = "{} {} vdepth {}".format(model['Featuremap'], model['Opt'], model['vdepth'])
            data[data_list] = test_benchmark.get_cost_list()

        if dataPath.__contains__("iris"):
            w = csv.writer(open("../../Data/Processed/iriscost1.csv", "w"))
        else:
            w = csv.writer(open("../../Data/Processed/winecost1.csv", "w"))
        for key, val in data.items():
            w.writerow([key, val])


if __name__ == "__main__":
    main()
