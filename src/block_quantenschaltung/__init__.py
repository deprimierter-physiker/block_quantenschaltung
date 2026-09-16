import numpy as np
import qiskit
from qiskit_aer import AerSimulator

"""
def apply_cnot(state, base, change):
    N = len(state)
    copy_state = np.copy(state)
    to_flip = []
    to_flip_0 = []
    k = 1
    for i in range(N):
        if i >= k * 2**base:
            to_flip.extend(range(i, i + 2**base))
            k += 2
    k = 1
    for i in range(N):
        if i >= k * 2**change:
            to_flip_0.extend(range(i, i + 2**change))
            k += 2
    print(to_flip_0, to_flip)
    for i in to_flip:
        if i not in to_flip_0:
            print("change accepted")
            copy_state[int(i + 2**change)] = state[i]
            copy_state[i] = state[int(i + 2**change)]
    return copy_state

strings = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]


def apply_single_qubit_gate(state, gate, qubit):
    N = np.log2(len(state))
    state_re = np.reshape(state, (2,) * N, order="F")
    sum_string = "A"
    sum_string += strings[qubit]
    sum_string += ","
    for i in range(N):
        sum_string += strings[i]
    sum_string += "->"
    altered_strings = np.copy(strings)
    altered_strings[qubit] = "A"
    for i in range(N):
        sum_string += altered_strings[i]
    product = np.einsum(sum_string, gate, state_re)
    return np.reshape(product, -1, order="F")

"""

class mock_simulate:
    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int, return_statevector: bool):
        self.circuit = circuit
        self.number_of_shots = number_of_shots
        self.return_statevector = return_statevector

    def perform_sim(self):
        simulator = AerSimulator(method="statevector")
        result = simulator.run(self.circuit, shots = self.number_of_shots).result()
        if self.return_statevector:
            return result.get_statevector(self.circuit)
        return result.get_counts(self.circuit)

class simulate:
    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int, return_statevector: bool):
        self.circuit = circuit
        self.number_of_shots = number_of_shots
        self.return_statevector = return_statevector
    def perform_sim(self):
        simulator = own_simulator(self.circuit, self.number_of_shots)
        result = simulator.run(self.circuit, shots = self.number_of_shots).result()
        if self.return_statevector:
            return result.get_statevector(self.circuit)
        return result.get_counts(self.circuit)

class own_simulator:
    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int):
        self.circuit = circuit
        self.number_of_shots = number_of_shots
        self.circuit = circuit
        self.state_vector = np.zeros([2] * self.circuit.num_qubits, dtype=complex)

    def single_qubit_gate(self, gate: np.ndarray, qubit_index: int, N: int, state_vector: np.ndarray) -> np.ndarray:

        state_tensor = np.reshape(state_vector, (2,) * N, order='F')

        strings = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

        qubit_string = []
        for i in range(N):
            qubit_string.append(strings[i])

        letter = strings[qubit_index]
        indicies = "I"+letter+","
        for i in range(len(qubit_string)):
            indicies += qubit_string[i] 

        new_qubit_string = [x for x in qubit_string if x != letter]
        new_indicies = "->I"
        for i in range(len(new_qubit_string)):
            new_indicies += new_qubit_string[i]
            

        result = np.einsum(f"{indicies+new_indicies}", gate, state_tensor)
        
        return result
    
    def apply_cnot(self, controll, target, state_vector):
        N = len(state_vector)
        copy_state = np.copy(state_vector)
        to_flip = []
        to_flip_0 = []
        k = 1
        for i in range(N):
            if i >= k * 2**controll:
                to_flip.extend(range(i, i + 2**controll))
                k += 2
        k = 1
        for i in range(N):
            if i >= k * 2**target:
                to_flip_0.extend(range(i, i + 2**target))
                k += 2
        print(to_flip_0, to_flip)
        for i in to_flip:
            if i not in to_flip_0:
                print("change accepted")
                copy_state[int(i + 2**target)] = state_vector[i]
                copy_state[i] = state_vector[int(i + 2**target)]
        return copy_state
        
def simulation_func(qc: qiskit.QuantumCircuit, number_of_shots: int, return_statevector: bool):
    qc.transpile() 
    state = np.zeros([2] * qc.num_qubits, dtype=complex)
    state[0] = 1.0
    
    def Hadamad_Gate() -> np.ndarray:
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    for information in qc.data:
        operation = information.operation
        name = operation.name  
    
        qubit_indices = [qc.find_bit(q).index for q in information.qubits]
        if name == "h":
            state = own_simulator.single_qubit_gate(own_simulator, Hadamad_Gate(), qubit_indices[0], qc.num_qubits, state)
        if name == "cx":
            state = own_simulator.apply_cnot(own_simulator, qubit_indices[0], qubit_indices[1], state)

    


