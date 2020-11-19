# Data Encoding

When we want to encode our classical data into quantum states we perform certain operations to help us work with the data in quantum circuits. One of the step is called data embedding which is the representation of classical data as a quantum state in Hilbert space via a quantum feature map. A feature map is the mathematical map that helps us embed our data into quantum states. It is usually a variational quantum circuit in which the parameters depend on the input which for our case is the classical data. We will need to define a variational quantum circuit before going any further. A variational quantum circuit is a quantum algorithm that depends on paramters and can there for be optimised by either classical or quantum methods.

## Data embedding
For data embedding we take out classical datapoint, x and make it into a set of paramters of a quantum gate in a quantum circuit hence creating out desirable quantum state.

x -> |phi(x)>

### Examples of data embedding methods

#### 1. Basis embedding
In this method we simply encode our data into binary strings. We convert each input to a computational basis state of a qubit system. For example x = 1001 is represented by a 4 qubit system as |1001> quantum state

#### 2. Amplitude embedding
We encode the data as amplitudes of a quantum state. A normalized classical N - dimensional datapoint x is represented by the amplitudes of a n-qubit quantum state. | phi(x) > as 


#### 3. Angle embedding

#### 4. Higher order embedding
It converts a 2D data point into a 3D data point

#### 5. Hamiltonian embedding
It encodes the classical data in the evolution of a quantum systems


Continuous variables are encoded using squeezing and displacement embedding.


## Feature maps
