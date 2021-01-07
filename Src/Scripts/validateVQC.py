#!/usr/bin/env python
# coding: utf-8
from qiskit import QuantumCircuit
from qiskit.aqua.components.optimizers import COBYLA, ADAM, SPSA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, ZFeatureMap, PauliFeatureMap
from qiskit.quantum_info import Statevector

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import warnings

warnings.filterwarnings("ignore")


class Benchmark:
    """
    Benchmarking different optimizers, featuremaps and depth of variational circuits
    """

    def __init__(self, optimizer, variational_depth, feature_map, X_train, X_test, Y_train, Y_test):
        """
        Initial function
        :param optimizer: The optimizer to benchmark
        :param variational_depth: The depth of the variational circuit
        :param feature_map: The featuremap that encodes data
        :param X_train: The x data for training
        :param X_test: The x data for testing
        :param Y_train: The y data for training
        :param Y_test: The y data for testing
        """
        self.optimizer = optimizer
        self.variational_depth = variational_depth
        self.feature_map = feature_map
        self.no_qubit = 4
        self.random_state = 42
        self.class_labels = ['yes', 'no']
        self.circuit = None
        self.var_form = RealAmplitudes(self.no_qubit, reps=self.variational_depth)
        self.sv = Statevector.from_label('0' * 4)
        self.X_train, self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test
        self.cost_list = []

    def prepare_circuit(self):
        """
        Prepares the circuit. Combines an encoding circuit, feature map, to a variational circuit, RealAmplitudes
        :return:
        """
        self.circuit = self.feature_map.combine(self.var_form)
        # circuit.draw(output='mpl')

    def get_data_dict(self, params, x):
        """
        Assign the params to the variational circuit and the data to the featuremap
        :param params: Parameter for training the variational circuit
        :param x: The data
        :return parameters:
        """
        parameters = {}
        for i, p in enumerate(self.feature_map.ordered_parameters):
            parameters[p] = x[i]
        for i, p in enumerate(self.var_form.ordered_parameters):
            parameters[p] = params[i]
        return parameters

    def assign_label(self, bit_string):
        """
        Based on the output from measurements assign no if it odd parity and yes if it is even parity
        :param bit_string: The bit string eg 00100
        :return class_label: Yes or No
        """
        hamming_weight = sum([int(k) for k in list(bit_string)])
        is_odd_parity = hamming_weight & 1
        if is_odd_parity:
            return self.class_labels[1]
        else:
            return self.class_labels[0]

    def return_probabilities(self, counts):
        """
        Calculates the probabilities of the class label after assigning the label from the bit string measured
        as output
        :type counts: dict
        :param counts: The counts from the measurement of the quantum circuit
        :return result: The probability of each class
        """
        shots = sum(counts.values())
        result = {self.class_labels[0]: 0, self.class_labels[1]: 0}
        for key, item in counts.items():
            label = self.assign_label(key)
            result[label] += counts[key] / shots
        return result

    def classify(self, x_list, params):
        """
        Assigns the x and params to the quantum circuit the runs a measurement to return the probabilities
        of each class
        :type params: List
        :type x_list: List
        :param x_list: The x data
        :param params: Parameters for optimizing the variational circuit
        :return probs: The probabilities
        """
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

    @staticmethod
    def mse_cost(probs, expected_label):
        """
        Calculates the mean squared error from the expected values and calculated values
        :type expected_label: List
        :type probs: List
        :param probs: The expected values
        :param expected_label: The real values
        :return mse: The mean squared error
        """
        p = probs.get(expected_label)
        actual, pred = np.array(1), np.array(p)
        mse = np.square(np.subtract(actual, pred)).mean()
        return mse

    def cost_function(self, X, Y, params, print_value=False):
        """
        This is the cost function and returns cost for optimization
        :type print_value: Boolean
        :type params: List
        :type Y: List
        :type X: List
        :param X: The x data
        :param Y: The label
        :param params: The parameters
        :param print_value: If you want values to be printed
        :return cost:
        """
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
        """
        Test the model based on x test and y test
        :type params: List
        :type Y: List
        :type X: List
        :param X: The x test set
        :param Y: The y test set
        :param params: The parameters
        :return:
        """
        accuracy = 0
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
        """
        Runs the whole code
        1. Prepares the circuit
        2. define the objective function
        3. Initialize the paramters
        4. Optimize the paramters by training the classifier
        :return:
        """
        self.prepare_circuit()
        # define objective function for training
        objective_function = lambda params: self.cost_function(self.X_train, self.Y_train, params, print_value=False)
        # randomly initialize the parameters
        np.random.seed(self.random_state)
        init_params = 2 * np.pi * np.random.rand(self.no_qubit * self.variational_depth * 2)
        # train classifier
        opt_params, value, _ = self.optimizer.optimize(len(init_params), objective_function, initial_point=init_params)
        # print results
        # print()
        # print('opt_params:', opt_params)
        # print('opt_value: ', value)

        self.test_model(self.X_test, self.Y_test, opt_params)

    def get_cost_list(self):
        """
        Return the cost list
        :return cost list:
        """
        return self.cost_list


def normalize_data(dataPath = "../../Data/Processed/iris_csv.csv"):
    """
    Normalizes the data
    """
    # Reads the data
    data = pd.read_csv(dataPath)
    data = shuffle(data, random_state=42)
    X, Y = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values, data['target'].values
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
    x_train, x_test, y_train, y_test = normalize_data()
    for fe in feature_maps:
        for i in [1, 3, 5]:
            for opt in optimizers:
                print("FE: {}\tDepth: {}\tOpt: {}".format(fe, i, opt))
                test_benchmark = Benchmark(optimizer=eval(opt), variational_depth=i, feature_map=eval(fe), X_train=x_train, X_test=x_test, Y_train=y_train, Y_test=y_test)
                test_benchmark.run()
                data_list = "{} {} vdepth {}".format(fe, opt, i)
                data[data_list] = test_benchmark.get_cost_list()

    w = csv.writer(open("../../Data/Processed/iriscost.csv", "w"))
    for key, val in data.items():
        w.writerow([key, val])


if __name__ == "__main__":
    main()
