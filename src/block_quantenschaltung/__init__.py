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

def pauli_x() -> numpy.ndarray:
    return numpy.array([[0, 1], [1, 0]], dtype=complex)


def pauli_y() -> numpy.ndarray:
    return numpy.array([[0, -1j], [1j, 0]], dtype=complex)


def pauli_z() -> numpy.ndarray:
    return numpy.array([[1, 0], [0, -1]], dtype=complex)

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
        result = own_simulator.simulation_func(self.circuit, self.number_of_shots)
        if self.return_statevector:
            return result[0]
        return result[1]

class own_simulator:
    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int):
        self.circuit = circuit
        # NOTE (dead state): neither self.circuit nor self.number_of_shots is ever read -
        # simulation_func takes the circuit and the shot count as arguments instead.
        self.number_of_shots = number_of_shots  
        #deleted self.state_vector variable since we did not need this one here bc we define it in aour sim_func
        """
        self.state_vector = np.zeros([2] * self.circuit.num_qubits, dtype=complex)
        self.state_vector[0] = 1
        """

    @staticmethod
    def single_qubit_gate(gate: np.ndarray, qubit_index: int, N: int, state_vector: np.ndarray) -> np.ndarray:

        state_tensor = np.reshape(state_vector, (2,) * N, order='F')
        qubit_axes = list(range(N)) #make the simulation for more then 26 qubits (before we used the alphabet)

        #Makes the labeling of our simulation in sync with the other methods
        axis = qubit_index
        out_label = N
        out_axes = qubit_axes.copy()
        out_axes[axis] = out_label

        result = np.einsum(gate, [out_label, axis], state_tensor, qubit_axes, out_axes)
        vec_res = np.reshape(result,-1, order='F')
        return vec_res
    
    @staticmethod
    def apply_cnot(controll, target, state_vector):
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
        for i in to_flip:
            if i not in to_flip_0:
                copy_state[int(i + 2**target)] = state_vector[i]
                copy_state[i] = state_vector[int(i + 2**target)]
        return copy_state

    @staticmethod
    def measurement_all(state_vector: np.ndarray, number_of_shots: int) -> dict[str, int]:
        """Sample the state `number_of_shots` times and tally the outcomes.

        The return value has the same structure as Aer's ``result.get_counts()``: keys are
        zero-padded bitstrings with qubit 0 as the RIGHTMOST character (Qiskit's little-endian
        convention, which is also the convention used by single_qubit_gate and apply_cnot),
        values are how often that outcome was drawn. Outcomes that never occurred are left
        out, exactly as Aer does.
        """
        N = len(state_vector)
        number_of_qubits = N.bit_length() - 1  # N is 2**number_of_qubits

        probabilities = np.abs(state_vector)**2

        # An explicit raise rather than `assert`: asserts are stripped by `python -O`, and then
        # the renormalisation below would silently rescale an unphysical state instead of
        # reporting that the gates lost norm.
        norm = np.sum(probabilities)
        if not np.isclose(norm, 1):
            raise ValueError(f"state vector is not normalised: sum(|amplitude|**2) = {norm}")
        probabilities /= norm

        # One vectorised draw instead of a python loop over the shots.
        samples = np.random.choice(N, size=number_of_shots, p=probabilities)
        indices, occurrences = np.unique(samples, return_counts=True)
        return {
            format(int(index), f"0{number_of_qubits}b"): int(count)
            for index, count in zip(indices, occurrences)
        }
    
        
        
    @staticmethod
    def simulation_func(qc: qiskit.QuantumCircuit, number_of_shots: int) -> list:
        # Aer's save_* instructions (save_statevector, save_probabilities, ...) are markers for
        # the Aer backend, not gates, and transpile() cannot translate them into the u/cx basis -
        # it raises TranspilerError at EVERY optimization level. Drop them: this simulator hands
        # the state back directly, so it does not need them.
        qc = qc.copy()
        qc.data = [datum for datum in qc.data if not datum.operation.name.startswith("save_")]

        # optimization_level=0 is required: from level 2 on, transpile() elides SWAP gates into a
        # final qubit permutation stored in qc.layout, which would silently permute the state.
        qc = qiskit.transpile(qc, basis_gates = ["u", "cx"], optimization_level=0) 
        state = np.zeros(2**qc.num_qubits, dtype=complex)
        state[0] = 1.0
        
        #def U_Gate(theta: float, phi: float, lam: float) -> np.ndarray:
        #    return np.array([[np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)], [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(lam+phi))*np.cos(theta/2)]], dtype=complex)

        for information in qc.data:
            operation = information.operation
            name = operation.name  
        
            qubit_indices = [qc.find_bit(q).index for q in information.qubits]

            if name == "u":
                #theta, phi, lam = operation.params
                matrix = operation.to_matrix()
                state = own_simulator.single_qubit_gate(matrix, qubit_indices[0], qc.num_qubits, state)
            elif name == "cx":
                state = own_simulator.apply_cnot(qubit_indices[0], qubit_indices[1], state)
            elif name == "measure":
                measurement_results = own_simulator.measurement_all(state, number_of_shots)
                return [state, measurement_results]
            elif name in ("barrier", "delay"):
                # Not physical operations - nothing to apply. measure_all() inserts a barrier,
                # so without this branch every measured circuit printed "ERROR: Unkown Gate".
                continue
            else:
                # NOTE: printing and carrying on means an unsupported instruction silently
                # produces a wrong state. Consider raising instead:
                #     raise NotImplementedError(f"gate not supported: {name}")
                print(f"ERROR: Unknown gate: {name}")

        return [state, None]


     





    


