import qiskit
from qiskit_aer import AerSimulator



class simulate:
    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int):
        self.circuit = circuit
        self.number_of_shots = number_of_shots
    def perform_sim(self):
        simulator = AerSimulator()
        result = simulator.run(self.circuit, shots = self.number_of_shots).result()
        return result
    def own_simulator(self):
        return 0
    
