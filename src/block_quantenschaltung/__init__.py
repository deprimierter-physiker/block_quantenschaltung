import qiskit
from qiskit_aer import AerSimulator



class simulate:
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
    def own_simulator(self):
        return 0
    
