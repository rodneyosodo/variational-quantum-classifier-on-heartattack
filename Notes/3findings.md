# An analysis of the variational quantum classifier using data

By now you should know how a variational quantum classifier works. The code for the previous series is at [Github repo](https://github.com/0x6f736f646f/variational-quantum-classifier-on-heartattack)


## Introduction
In binary classification, let's say labelling if someone is likely to have a heart attack or not, we would build a function that takes in the information about the patient and gives results aligned to the reality. E.g, $f(information) = P(hear attack = YES)$. This probabilistic classification is well suited for quantum computing and we would like to build a quantum state that, when measured and post processed, returns $P(hear attack = YES)$. By optimising the circuits, you then find parameters that will give the closest probability to the reality based on training data. 

## Problem statement
Given a dataset about patients' information, can we predict if the patient is likely to have a heart attack or not? This is a binary classification problem, with a real input vector ${x}$ and a binary output ${y}$ in $\{0, 1\}$. We want to build a quantum circuit whose output is the quantum state: 
![](../Notes/findings/math-7.png)

## Implementation
1. We initialise our circuit in the zero state (all qubits in state zero)
```python
self.sv = Statevector.from_label('0' * self.no_qubit)
```
2. We use a feature map such as, `ZZFeaturemap, ZFeaturemap` or `PauliFeaturemap` and choose the number of qubits based on the input dimension of the data and how many repetitions (i.e. the circuit depth) we want. We use 1, 3, 5.

3. We choose the variational form as `RealAmplitudes` and specify the number of qubits as well as how many repetitions we want. We use 1, 2, 4 to have models with an increasing number of trainable parameters.

4. We then combine our feature map to the variational circuit.
`ZZfeaturemap and RealAmplitudes both with a depth of 1`
```python
def prepare_circuit(self):
    """
    Prepares the circuit. Combines an encoding circuit, feature map, to a variational circuit, RealAmplitudes
    :return:
    """
    self.circuit = self.feature_map.combine(self.var_form)
```
5. We create a function that associates the parameters of the feature map with the data and the parameters of the variational circuit with the parameters passed. This is to ensure in Qiskit that the right variables in the circuit are associated with the right quantities.
```python
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

```

![](../Output/Figures/parameterisedcircuit.png)


6. We create another function that checks the parity of the bit string passed. Hence if the parity is even it returns a yes and if the parity is odd it returns a no @Rodney please explain why this choice and that other options exist
```python
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
```
1.  We create another function that returns the probability distribution over the model classes. @Rodney why?? You need to explain these steps to help people understand more.
```python
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
```
8.   Finally, we create a function that classifies our data. It takes in data and parameters. For every data point in the dataset we assign the parameters to the feature map and the parameters to the variational circuit. We then evolve our system and store the quantum circuit. We store the circuits so as to run them at once at the end. We measure each circuit and return the probabilities based on the bit string and class labels  @Rodney please explain more why we store multiple circuits
```python
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
```

## Results
Data classification was performed by using the implemented version of VQC in IBM’s framework and executed on the provider simulator 
```python
qiskit==0.23.1
qiskit-aer==0.7.1
qiskit-aqua==0.8.1
qiskit-ibmq-provider==0.11.1
qiskit-ignis==0.5.1
qiskit-terra==0.16.1
```

Every combination of the experiments were executed with 1024 shots, using the implemented version of the optimizers. We conducted tests with different feature map depths, variational depths and optimizers. In each case, we compared loss values. Our best configs were 
```python
ZFeatureMap(4, reps=2) SPSA(max_trials=50) vdepth 5 : Cost: 0.13492279429495616
ZFeatureMap(4, reps=2) SPSA(max_trials=50) vdepth 3 : Cost: 0.13842958846394343
ZFeatureMap(4, reps=2) COBYLA(maxiter=50) vdepth 3 : Cost: 0.14097642258192988
ZFeatureMap(4, reps=2) SPSA(max_trials=50) vdepth 1 : Cost: 0.14262128997684975
ZFeatureMap(4, reps=1) COBYLA(maxiter=50) vdepth 1 : Cost: 0.1430145495411656
ZZFeatureMap(4, reps=1) SPSA(max_trials=50) vdepth 5 : Cost: 0.14359757088670677
ZFeatureMap(4, reps=2) COBYLA(maxiter=50) vdepth 5 : Cost: 0.1460568741051525
ZFeatureMap(4, reps=1) SPSA(max_trials=50) vdepth 3 : Cost: 0.14830080135566964
ZFeatureMap(4, reps=1) SPSA(max_trials=50) vdepth 5 : Cost: 0.14946706294763648
ZFeatureMap(4, reps=1) COBYLA(maxiter=50) vdepth 3 : Cost: 0.15447151389989414
```

@Rodney you need to elaborate more here. This is the most important part of the entire project
## Questions
#### 1. Does increasing variational depth increase convergence?
- When increasing vdepth on `ZZFeatureMap(4, reps=1) SPSA(max_trials=50)`, `ZZFeatureMap(4, reps=2) SPSA(max_trials=50)`, `ZZFeatureMap(4, reps=2) ADAM(maxiter=50)` and `PauliFeatureMap(4, reps=2) ADAM(maxiter=50)` increases the convergence. The rest it doesn't achieve considerable increase in converegence. In some it actualyy reduces convergences almost linearly

#### 2. Does increasing featuremap depth increase convergence?
- When increasing fdepth on `ZZFeatureMap ADAM (maxiter=50) vdepth 5` and `PauliFeatureMap ADAM(maxiter=50) vdepth 5` increases the convergence. The rest it doesn't achieve considerable increase in converegence. In some it actualyy reduces convergences almost linearly


@Rodney, you should also start with a nice motivation in the first notebook as to why we used heart attack data
## Conclusion
Heart attack is a major concern in public health, therefore several research efforts have been conducted including topics that are addressed using statistics and data mining. Every year more data is becoming available from the increased diagnosis rate. The data availability has lead to the emergence of machine learning methods, which are nowadays an extremely valuable tool for healthcare professionals to make and understand diagnoses and mitigate risks. In general, VQC results demonstrate that it is a promising technique when quantum devices grow in its capabilities, attending the future necessities of the healthcare system.
