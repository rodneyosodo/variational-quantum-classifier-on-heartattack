# Data Encoding, state preparation

When we want to encode our classical data into quantum states we perform certain operations to help us work with the data in quantum circuits. One of the step is called data embedding which is the representation of classical data as a quantum state in Hilbert space via a quantum feature map. A feature map is the mathematical map that helps us embed our data into quantum states. It is usually a variational quantum circuit in which the parameters depend on the input which for our case is the classical data. We will need to define a variational quantum circuit before going any further. A variational quantum circuit is a quantum algorithm that depends on paramters and can there for be optimised by either classical or quantum methods.

## Data embedding
For data embedding we take out classical datapoint, x and make it into a set of paramters of a quantum gate in a quantum circuit hence creating out desirable quantum state.

x -> |phi(x)>

### Examples of data embedding methods

#### 1. Basis embedding
In this method we simply encode our data into binary strings. We convert each input to a computational basis state of a qubit system. For example x = 1001 is represented by a 4 qubit system as |1001> quantum state
- amplitude vectors become sparse
- most freedom to do computation
- schemes not efficient


#### 2. Amplitude embedding
We encode the data as amplitudes of a quantum state. A normalized classical N - dimensional datapoint x is represented by the amplitudes of a n-qubit quantum state. | phi(x) > as 

- simple, intuitive
- inexpensive

#### 3. Angle embedding
Here, we use the so-called angle encoding 
|xi = On i=1 cos(xi)|0i + sin(xi)|1i 

where x = (x0, ...xN )T

Practically, this amounts to using the input data, x, as angles in
a unitary quantum gate. We take the state preparation circuit as the unitary gate

Ry(θ) = cos(θ/2) -sin(θ/2) sin(θ/2) cos(θ/2)!

#### 4. Higher order embedding
It converts a 2D data point into a 3D data point

#### 5. Hamiltonian embedding
It encodes the classical data in the evolution of a quantum systems

Continuous variables are encoded using squeezing and displacement embedding.


## Feature maps
Feature maps allow you to map data into a higher dimensionsal Hilbert Space. This allows you to perform computation over non-linear basic functions.
This encodes our classical data xi into quantum states |φ(xi)>. Using parameterized quantum circuits implies using exponential number of functions with respect to the no of qubit from the parameter circuit.

## Implementation
1. We initialize our circuit with Zero state and multipy it with time, number of qubits
2. We use a higher order featiremap, `ZZFeaturemap` and specify the number of qubits and also how many repetions we want.
3. We specify the variational from as `RealAmplitude` and specify the number of qubits and also how many repetions we want.
4. We then combine our featremap to the variational quantum circuit.


5. Wecreate a function to that associates the parameters of the featuremap with the data and the parameters of the variational circuit with the parameters passed


6. We create another functions that checks the parity of the bitsring passed. hence if the parity is even it returns a yes and if the parity os odd it returns a no


7.  We create another functions that returns the probabiltiy distribution over the model classes


8.   We create another functions that classifies our data. It takes in data and paramters. For every datapointin the dataset we assign the paramters to the featuremap and the parameters to the variational circuit. We then evolve our system and store the quantum circuit. We then measure each circuit and return the probabilities based on the bit string and class labels
