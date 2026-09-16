import numpy as np
import qiskit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt  # BUG (minor): imported but never used

# NOTE: the large triple-quoted block below is an older draft (Pauli matrices, an earlier
# apply_cnot / apply_single_qubit_gate). It is dead and partly broken - e.g. it calls
# numpy.* while this module imports numpy as np, and np.reshape((2,)*N) with a float N
# from np.log2. Keep it out of the way (or delete it) so it is not confused with the
# live implementation further down.

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
        # BUG (fatal): this mimics the AerSimulator API of mock_simulate, but own_simulator
        # has no .run() and never produces a result object -> AttributeError before any
        # gate math runs. The real gate loop lives in simulation_func() and is never called.
        # FIX: return simulation_func(self.circuit, self.number_of_shots)
        simulator = own_simulator(self.circuit, self.number_of_shots)
        result = simulator.simulation_func(self.circuit, shots = self.number_of_shots).result()
        if self.return_statevector:
            return result.get_statevector(self.circuit)
        return result.get_counts(self.circuit)
        #TODO: match syntax

class own_simulator:
    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int):
        self.circuit = circuit
        self.number_of_shots = number_of_shots
        self.circuit = circuit  # BUG (harmless): duplicate assignment of self.circuit
        # BUG: dead state. Nothing ever reads self.state_vector - simulation_func() builds
        # its own independent flat `state` array instead. The class-based and the
        # function-based design were never merged.
        self.state_vector = np.zeros([2] * self.circuit.num_qubits, dtype=complex)
        self.state_vector[0] = 1

    def single_qubit_gate(self, gate: np.ndarray, qubit_index: int, N: int, state_vector: np.ndarray) -> np.ndarray:

        state_tensor = np.reshape(state_vector, (2,) * N, order='F')

        # BUG (limit): hardcoded 26 letters -> IndexError for circuits with more than
        # 26 qubits. FIX: generate labels programmatically, or use einsum's list-of-ints API.
        strings = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

        qubit_string = []
        for i in range(N):
            qubit_string.append(strings[i])

        # BUG (confirmed by experiment): the N-1-qubit_index reversal picks the WRONG axis.
        # np.reshape(state, (2,)*N, order='F') already maps tensor axis i -> qubit i
        # (i.e. bit i of the flat index) - the same convention apply_cnot() assumes below.
        # So every gate lands on qubit N-1-q instead of q (only correct when q == N-1-q).
        # FIX: letter = strings[qubit_index]
        letter = strings[N - 1 - qubit_index]
        indicies = "I"+letter+","
        for i in range(len(qubit_string)):
            indicies += qubit_string[i] 

        # BUG (confirmed by experiment, separate from the one above - fixing only the
        # `letter` line is NOT enough): the output spec always puts the new label "I" FIRST
        # instead of back into the slot the contracted axis occupied. Every gate on a qubit
        # other than qubit 0 therefore PERMUTES the qubit <-> bit-position mapping, and the
        # order='F' reshape below silently bakes that permutation into the state vector,
        # corrupting all following gates and the measurement (unphysical results).
        # FIX: new_qubit_string = qubit_string.copy(); new_qubit_string[qubit_index] = "I"
        new_qubit_string = [x for x in qubit_string if x != letter]
        new_indicies = "->I"
        for i in range(len(new_qubit_string)):
            new_indicies += new_qubit_string[i]
            
        #TODO:wrong axis
        result = np.einsum(f"{indicies+new_indicies}", gate, state_tensor)
        vec_res = np.reshape(result,-1, order='F')
        return vec_res
    
    def apply_cnot(self, controll, target, state_vector):
        # NOTE (checked, NOT a bug): the k / k += 2 threshold loops below look like a
        # fragile reimplementation of "is bit b of i set", but they do produce the same
        # index sets as (i >> b) & 1, and the swap is a correct CNOT. Obscure and O(N^2)
        # because of the `i not in to_flip_0` membership test, but correct - don't chase it.
        # (`controll` is a typo for `control`.)
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
        print(to_flip_0, to_flip)  # BUG (noise): leftover debug print, floods output per gate
        for i in to_flip:
            if i not in to_flip_0:
                print("change accepted")  # BUG (noise): leftover debug print
                copy_state[int(i + 2**target)] = state_vector[i]
                copy_state[i] = state_vector[int(i + 2**target)]
        return copy_state

    def measurement_all(self, state_vector: np.ndarray, number_of_shots: int) -> str:
        # BUG: number_of_shots is accepted but ignored - exactly ONE sample is drawn and a
        # single bitstring returned, while the Aer reference path returns a counts dict of
        # size `shots`. The two results are structurally incomparable, so the tests can
        # never pass for the counts case even once the gate math is fixed.
        # FIX: sample number_of_shots times and tally into a dict[str, int] like get_counts().
        N = len(state_vector)
        probabilities = np.abs(state_vector)**2
        # NOTE: this renormalisation hides loss of norm. If the gate math above is wrong the
        # state is silently rescaled here instead of failing loudly - consider asserting
        # np.isclose(np.sum(probabilities), 1) so unphysical states are caught.
        probabilities /= np.sum(probabilities)
        measurement_result = np.random.choice(range(N), p=probabilities) #repeat N times
        return format(measurement_result, f"0{int(np.log2(N))}b")
    
        
        
# BUG (contract): annotated -> np.ndarray, but returns the 2-tuple (state, measurement_results)
# on the `measure` branch and a bare array otherwise. Callers cannot rely on the return shape.
# FIX: pick one contract, e.g. always return (state, counts_or_None), and fix the annotation.
def simulation_func(qc: qiskit.QuantumCircuit, number_of_shots: int) -> np.ndarray:
    qc = qiskit.transpile(qc, basis_gates = ["u", "cx"]) 
    state = np.zeros(2**qc.num_qubits, dtype=complex)
    state[0] = 1.0
    
    #def U_Gate(theta: float, phi: float, lam: float) -> np.ndarray:
    #    return np.array([[np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)], [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(lam+phi))*np.cos(theta/2)]], dtype=complex)

    for information in qc.data:
        operation = information.operation
        name = operation.name  
    
        qubit_indices = [qc.find_bit(q).index for q in information.qubits]
        # BUG (design): the three calls below pass the CLASS own_simulator as `self` instead
        # of an instance. It only "works" because none of these methods read self - which is
        # the same reason self.state_vector in __init__ is dead code.
        # FIX: make them @staticmethod, or build one instance and call the bound methods.
        if name == "u":
            #theta, phi, lam = operation.params
            matrix = operation.to_matrix()
            state = own_simulator.single_qubit_gate(own_simulator, matrix, qubit_indices[0], qc.num_qubits, state)
        if name == "cx":
            state = own_simulator.apply_cnot(own_simulator, qubit_indices[0], qubit_indices[1], state)
        if name == "measure":
            # BUG (semantics): returns immediately on the FIRST measure instruction, so any
            # gates after it are skipped. measure_all() emits one `measure` per qubit, so a
            # circuit ending in measure_all effectively measures after the first one only.
            # Also: a real measurement should collapse the state, not leave it untouched.
            measurement_results = own_simulator.measurement_all(own_simulator, state, number_of_shots)
            return state, measurement_results

    return state


     





    


