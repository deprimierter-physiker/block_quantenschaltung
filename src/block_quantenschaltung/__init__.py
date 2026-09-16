import qiskit
from qiskit_aer import AerSimulator
import numpy as np



class simulate:
    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int):
        self.circuit = circuit
        self.number_of_shots = number_of_shots

    def perform_sim(self):
        simulator = AerSimulator()
        result = simulator.run(self.circuit, shots = self.number_of_shots).result()
        return result

class own_simulator:
    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int):
        self.circuit = circuit
        self.number_of_shots = number_of_shots
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
    

        
