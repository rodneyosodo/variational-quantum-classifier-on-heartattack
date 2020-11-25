#!/usr/bin/env python
# coding: utf-8
from qiskit import QuantumCircuit
from qiskit.aqua.components.optimizers import COBYLA, ADAM, SPSA, L_BFGS_B
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, ZFeatureMap, PauliFeatureMap
from qiskit.quantum_info import Statevector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import json
import csv
import warnings
warnings.filterwarnings("ignore")

class Benchmark:
    def __init__(self, optimizer, variational_depth, feature_map, X_train, X_test, Y_train, Y_test):
        self.optimizer = optimizer
        self.variational_depth = variational_depth
        self.feature_map = feature_map
        self.no_qubit = 4
        self.random_state = 42
        self.class_labels = ['yes', 'no']
        self.circuit = None
        self.sv = Statevector.from_label('0' * 4)
        self.X_train, self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test
        self.cost_list = []
    

    def prepare_circuit(self): 
        self.var_form = RealAmplitudes(self.no_qubit, reps=self.variational_depth)
        self.circuit = self.feature_map.combine(self.var_form)
        # circuit.draw(output='mpl')
    
    def get_data_dict(self, params, x):
        parameters = {}
        for i, p in enumerate(self.feature_map.ordered_parameters):
            parameters[p] = x[i]
        for i, p in enumerate(self.var_form.ordered_parameters):
            parameters[p] = params[i]
        return parameters

    def assign_label(self, bit_string):
        hamming_weight = sum([int(k) for k in list(bit_string)])
        is_odd_parity = hamming_weight & 1
        if is_odd_parity:
            return self.class_labels[1]
        else:
            return self.class_labels[0]

    def return_probabilities(self, counts):
        shots = sum(counts.values())
        result = {self.class_labels[0]: 0, self.class_labels[1]: 0}
        for key, item in counts.items():
            label = self.assign_label(key)
            result[label] += counts[key]/shots
        return result

    def classify(self, x_list, params):
        qc_list = []
        for x in x_list:
            circ_ = self.circuit.assign_parameters(self.get_data_dict(params, x))
            qc = self.sv.evolve(circ_)
            qc_list += [qc]
            probs = []
        for qc in qc_list:
            counts = qc.to_counts()
            prob = self.return_probabilities(counts)
            probs += [prob]
        return probs

    def mse_cost(self, probs, expected_label):
        p = probs.get(expected_label)
        actual, pred = np.array(1), np.array(p)
        return np.square(np.subtract(actual,pred)).mean()

    def cost_function(self, X, Y, params, shots=100, print_value=False):
        # map training input to list of labels and list of samples
        cost = 0
        training_labels = []
        training_samples = []
        for sample in X:
            training_samples += [sample]
        for label in Y:
            if label == 0:
                training_labels += [self.class_labels[0]]
            elif label == 1:
                training_labels += [self.class_labels[1]]
                
        probs = self.classify(training_samples, params)
        # evaluate costs for all classified samples
        for i, prob in enumerate(probs):
            cost += self.mse_cost(prob, training_labels[i])
        cost /= len(training_samples)
        # print resulting objective function
        if print_value:
            print('%.4f' % cost)
        # return objective value
        self.cost_list.append(cost)
        return cost

    def test_model(self, X, Y, params):
        accuracy = 0
        training_labels = []
        training_samples = []
        for sample in X:
            training_samples += [sample]
        probs = self.classify(training_samples, params)
        for i, prob in enumerate(probs):
            if (prob.get('yes') >= prob.get('no')) and (Y[i] == 0):
                accuracy += 1
            elif (prob.get('no') >= prob.get('yes')) and (Y[i] == 1):
                accuracy += 1
        accuracy /= len(Y)
        print("Test accuracy: {}".format(accuracy))
    
    def run(self):
        self.prepare_circuit()
        # define objective function for training
        objective_function = lambda params: self.cost_function(self.X_train, self.Y_train, params, print_value=False)
        # randomly initialize the parameters
        np.random.seed(self.random_state)
        init_params = 2 * np.pi * np.random.rand(self.no_qubit * (self.variational_depth) * 2)
        # train classifier
        opt_params, value, _ = self.optimizer.optimize(len(init_params), objective_function, initial_point=init_params)
        # print results
        # print()
        # print('opt_params:', opt_params)
        # print('opt_value: ', value)

        self.test_model(self.X_test, self.Y_test, opt_params)
    
    def get_cost_list(self):
        return self.cost_list


def normalize_data(DATA_PATH = "../../Data/Processed/data.csv"):
    """
    Normalizes the data
    """
    # Reads the data
    data = pd.read_csv(DATA_PATH)
    data = shuffle(data, random_state=42)
    X, Y = data[['sex', 'cp', 'exang', 'oldpeak']].values, data['num'].values
    # normalize the data
    scaler = MinMaxScaler(feature_range=(-2 * np.pi, 2 * np.pi))
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    return X_train, X_test, Y_train, Y_test


def main():
    data = {}
    feature_maps = ['ZZFeatureMap(4, reps=1)', 'ZZFeatureMap(4, reps=2)', 'ZZFeatureMap(4, reps=4)',
                    'ZFeatureMap(4, reps=1)', 'ZFeatureMap(4, reps=2)', 'ZFeatureMap(4, reps=4)',
                    'PauliFeatureMap(4, reps=1)', 'PauliFeatureMap(4, reps=2)', 'PauliFeatureMap(4, reps=4)']
    optimizers = ["COBYLA(maxiter=50)", "SPSA(max_trials=50)", "ADAM(maxiter=50)"]
    X_train, X_test, Y_train, Y_test = normalize_data()
    for fe in feature_maps:
        for i in [1, 3, 5]:
            for opt in optimizers:
                print("FE: {}\tDepth: {}\tOpt: {}".format(fe, i, opt))
                test_benchmark = Benchmark(optimizer=eval(opt),
                                            variational_depth=i,
                                            feature_map=eval(fe),
                                            X_train=X_train, X_test=X_test,
                                            Y_train=Y_train, Y_test=Y_test)
                test_benchmark.run()
                data_list = "{} {} vdepth {}".format(fe, opt, i)
                data[data_list] = test_benchmark.get_cost_list()
    

    w = csv.writer(open("../../Data/Processed/costs.csv", "w"))
    for key, val in data.items():
        w.writerow([key, val])

if __name__ == "__main__":
    main()