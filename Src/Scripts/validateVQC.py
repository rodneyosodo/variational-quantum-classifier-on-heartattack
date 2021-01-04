#!/usr/bin/env python
# coding: utf-8
from qiskitBenchmarking import Benchmark, normalize_data
import csv


import numpy as np
import pandas as pd
from qiskit.aqua.components.optimizers import COBYLA, ADAM, SPSA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, ZFeatureMap, PauliFeatureMap

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import warnings

warnings.filterwarnings("ignore")


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
    feature_maps = ['ZZFeatureMap(4, reps=1)', 'ZZFeatureMap(4, reps=2)', 'ZZFeatureMap(4, reps=4)',
                    'ZFeatureMap(4, reps=1)', 'ZFeatureMap(4, reps=2)', 'ZFeatureMap(4, reps=4)',
                    'PauliFeatureMap(4, reps=1)', 'PauliFeatureMap(4, reps=2)', 'PauliFeatureMap(4, reps=4)']
    optimizers = ["COBYLA(maxiter=50)", "SPSA(max_trials=50)", "ADAM(maxiter=50)"]
    data = ["../../Data/Processed/iris_csv.csv", "../../Data/Processed/winedata.csv"]
    for dataPath in data:
        x_train, x_test, y_train, y_test = normalize_data(dataPath = dataPath)
        for fe in feature_maps:
            for i in [1, 3, 5]:
                for opt in optimizers:
                    print("FE: {}\tDepth: {}\tOpt: {}".format(fe, i, opt))
                    test_benchmark = Benchmark(optimizer=eval(opt), variational_depth=i, feature_map=eval(fe), X_train=x_train, X_test=x_test, Y_train=y_train, Y_test=y_test)
                    test_benchmark.run()
                    data_list = "{} {} vdepth {}".format(fe, opt, i)
                    data[data_list] = test_benchmark.get_cost_list()

        if dataPath.__contains__("iris"):
            w = csv.writer(open("../../Data/Processed/iriscost.csv", "w"))
        else:
            w = csv.writer(open("../../Data/Processed/winecost.csv", "w"))
        for key, val in data.items():
            w.writerow([key, val])


if __name__ == "__main__":
    main()
