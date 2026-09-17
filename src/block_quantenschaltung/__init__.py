"""A small state-vector simulator for quantum circuits.

The package provides two interchangeable back ends for running a
:class:`qiskit.QuantumCircuit`, plus a reference back end built on Qiskit Aer that the
own implementations are validated against:

* :class:`mock_simulate` -- delegates to Qiskit Aer's ``statevector`` method. Used as the
  ground truth in the test suite.
* :class:`simulate` -- the own simulator, applying single-qubit gates with
  :func:`numpy.einsum` (see :meth:`own_simulator.single_qubit_gate`).
* :class:`simulate_no_einsum` -- the own simulator, applying single-qubit gates with an
  explicit index loop instead (see :func:`apply_U`).

All three share the same constructor signature and expose a single ``perform_sim()``
method, so they can be swapped without touching the calling code.

Conventions:
    Amplitudes are stored in a flat, little-endian state vector of length ``2**num_qubits``:
    the amplitude of basis state :math:`|b_{n-1} \\dots b_1 b_0\\rangle` sits at index
    :math:`\\sum_k b_k 2^k`, so qubit 0 is the least significant bit. This matches Qiskit's
    own ordering, which is why measurement bitstrings have qubit 0 as their rightmost
    character.

Example:
    Simulate a Bell state and read back the state vector::

        import block_quantenschaltung as qs

        circuit = qs.qiskit.QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        state = qs.simulate(circuit, 1000, True).perform_sim()
"""

import numpy as np
import numpy.typing as npt
import qiskit  # type: ignore[import-untyped]
from qiskit_aer import AerSimulator  # type: ignore[import-untyped]
from numba import njit

#: A flat, little-endian state vector of complex amplitudes, of length ``2**num_qubits``.
StateVector = npt.NDArray[np.complex128]

#: Return value of the own simulators: ``[state_vector, counts_or_None]``.
SimulationResult = list[StateVector | dict[str, int] | None]


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

@njit
def apply_U(
    matrix: npt.NDArray[np.complex128],
    qubit_index: int,
    num_qubits: int,
    state: StateVector,
) -> StateVector:
    """Apply a single-qubit operator to one qubit of a state vector, without einsum.

    A single-qubit gate only ever mixes pairs of amplitudes whose basis-state bitstrings
    are identical everywhere except at ``qubit_index``. Those partners are exactly the
    index pairs ``(i, i + 2**qubit_index)``, which this function walks explicitly: the
    outer loop steps over the blocks above the target bit, the inner loop over the
    ``2**qubit_index`` indices below it.

    This is the loop-based counterpart of :meth:`own_simulator.single_qubit_gate` and is
    used by :class:`own_simulator_no_einsum`. Both produce the same state; keeping the two
    around allows the einsum and the hand-rolled index arithmetic to be compared.

    Args:
        matrix: The 2x2 unitary to apply, as a complex array of shape ``(2, 2)``.
        qubit_index: Index of the qubit the gate acts on, with qubit 0 the least
            significant bit of the basis-state index.
        num_qubits: Total number of qubits in the register. ``state`` must therefore have
            ``2**num_qubits`` entries.
        state: The state vector to act on. It is not modified; a new array is returned.

    Returns:
        A new state vector of the same shape as ``state``, with ``matrix`` applied to
        ``qubit_index``.
    """
    return_state = np.copy(state)
    for left in range(0, 2**num_qubits, 2**(qubit_index+1)):
        for right in range(2**qubit_index):
            low_state_idx = left+right
            high_state_idx = low_state_idx + 2**qubit_index
            small_vec = np.array([state[low_state_idx], state[high_state_idx]])
            res = np.dot(matrix, small_vec)
            return_state[low_state_idx] = res[0]
            return_state[high_state_idx] = res[1]
    return return_state

def apply_CNOT_clean(control: int, target: int, state_vector: np.ndarray) -> np.ndarray:
    return_state = np.copy(state_vector)
    num_qubits = len(state_vector)
    c_1_states_idx = []
    for left in range(0, 2**num_qubits, 2**(control+1)):
        for right in range(2**control):
            c_1_states_idx.append(left+right+ 2**control)
    contracted_state = np.array([state_vector[i] for i in c_1_states_idx])
    correction = 0
    if control < target:
        correction = 1
    for left in range(0, len(contracted_state), 2**(target-correction+1)):
            for right in range(2**target-correction):
                low_state_idx = left+right
                high_state_idx = low_state_idx + 2**(target-correction)
                return_state[low_state_idx + 2**control] = contracted_state[high_state_idx]
                return_state[high_state_idx + 2**control] = contracted_state[low_state_idx]
    return return_state


class mock_simulate:
    """Run a circuit on Qiskit Aer's state-vector simulator.

    This is the reference implementation: it wraps :class:`qiskit_aer.AerSimulator` behind
    the same interface as :class:`simulate` and :class:`simulate_no_einsum`, so the test
    suite can compare the own simulators against it by swapping the class name alone.

    Attributes:
        circuit: The circuit to simulate.
        number_of_shots: Number of measurement repetitions to request from Aer.
        return_statevector: Whether :meth:`perform_sim` returns the state vector rather
            than the measurement counts.

    Example:
        ::

            circuit = qiskit.QuantumCircuit(2)
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.save_statevector()

            state = mock_simulate(circuit, 1000, True).perform_sim()
    """

    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int, return_statevector: bool) -> None:
        """Store the simulation parameters.

        Args:
            circuit: The circuit to simulate. To retrieve a state vector it must carry a
                ``save_statevector()`` instruction, which Aer needs as an explicit marker.
            number_of_shots: Number of measurement repetitions.
            return_statevector: If ``True``, :meth:`perform_sim` returns the final state
                vector; if ``False``, it returns the measurement counts.
        """
        self.circuit = circuit
        self.number_of_shots = number_of_shots
        self.return_statevector = return_statevector

    def perform_sim(self) -> qiskit.quantum_info.Statevector | dict[str, int]:
        """Run the circuit on Aer and return the requested result.

        Returns:
            The final :class:`qiskit.quantum_info.Statevector` if ``return_statevector`` is
            ``True``, otherwise the measurement counts as a dict mapping zero-padded
            bitstrings to how often they were observed.

        Raises:
            qiskit.exceptions.QiskitError: If the state vector was requested but the
                circuit contains no ``save_statevector()`` instruction.
        """
        simulator = AerSimulator(method="statevector")
        result = simulator.run(self.circuit, shots = self.number_of_shots).result()
        if self.return_statevector:
            return result.get_statevector(self.circuit)
        return result.get_counts(self.circuit)


class simulate:
    """Run a circuit on the own simulator, using einsum for single-qubit gates.

    A thin front end over :meth:`own_simulator.simulation_func`, mirroring the interface of
    :class:`mock_simulate`. Single-qubit gates go through
    :meth:`own_simulator.single_qubit_gate`, which contracts the gate into the state with
    :func:`numpy.einsum`.

    Attributes:
        circuit: The circuit to simulate.
        number_of_shots: Number of measurement repetitions.
        return_statevector: Whether :meth:`perform_sim` returns the state vector rather
            than the measurement counts.
    """

    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int, return_statevector: bool) -> None:
        """Store the simulation parameters.

        Args:
            circuit: The circuit to simulate. Unlike :class:`mock_simulate` it needs no
                ``save_statevector()`` instruction, since the state is handed back
                directly; any ``save_*`` instruction present is stripped before running.
            number_of_shots: Number of measurement repetitions.
            return_statevector: If ``True``, :meth:`perform_sim` returns the final state
                vector; if ``False``, it returns the measurement counts.
        """
        self.circuit = circuit
        self.number_of_shots = number_of_shots
        self.return_statevector = return_statevector

    def perform_sim(self) -> StateVector | dict[str, int] | None:
        """Run the circuit on the own simulator and return the requested result.

        Returns:
            The final state vector if ``return_statevector`` is ``True``. Otherwise the
            measurement counts, or ``None`` if the circuit contained no ``measure``
            instruction and therefore produced no counts.
        """
        result = own_simulator.simulation_func(self.circuit, self.number_of_shots)
        if self.return_statevector:
            return result[0]
        return result[1]


class own_simulator:
    """State-vector simulator applying single-qubit gates with :func:`numpy.einsum`.

    The class is a namespace of static methods; :meth:`simulation_func` is the entry point
    and drives the whole simulation. Use :class:`simulate` for the usual front end.

    Supported instructions are ``u`` and ``cx`` -- every circuit is transpiled into that
    basis first -- plus ``measure``, ``barrier`` and ``delay``.

    Attributes:
        circuit: The circuit passed at construction.
        number_of_shots: The shot count passed at construction.
    """

    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int) -> None:
        """Store the circuit and shot count.

        Note:
            Instantiating this class is not required: :meth:`simulation_func` is a static
            method that takes the circuit and the shot count as arguments, and neither
            attribute set here is ever read.

        Args:
            circuit: The circuit to simulate.
            number_of_shots: Number of measurement repetitions.
        """
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
    def single_qubit_gate(
        gate: npt.NDArray[np.complex128],
        qubit_index: int,
        N: int,
        state_vector: StateVector,
    ) -> StateVector:
        """Apply a single-qubit operator to one qubit, via an einsum contraction.

        The flat state vector is reshaped into an ``N``-dimensional tensor with one axis of
        length 2 per qubit, the gate is contracted against the axis belonging to
        ``qubit_index``, and the result is flattened again. Fortran ordering is used
        throughout so that axis ``k`` corresponds to bit ``k`` of the flat index, keeping
        the little-endian convention of the rest of the package.

        :func:`numpy.einsum` is called with the integer-label form rather than the
        subscript-string form, which lifts the 26-qubit ceiling a letter-based alphabet
        would impose.

        Args:
            gate: The 2x2 unitary to apply, as a complex array of shape ``(2, 2)``.
            qubit_index: Index of the qubit the gate acts on, qubit 0 being the least
                significant bit.
            N: Total number of qubits in the register.
            state_vector: The state vector to act on. It is not modified.

        Returns:
            A new state vector of the same shape, with ``gate`` applied to ``qubit_index``.
        """
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
    def apply_cnot(controll: int, target: int, state_vector: StateVector) -> StateVector:
        """Apply a controlled-NOT gate to the state vector.

        The amplitude pairs to exchange are those whose basis-state bitstring has a 1 at
        ``controll``; for each of them the ``target`` bit is flipped by swapping the
        amplitude with the one at ``index + 2**target``.

        Args:
            controll: Index of the control qubit, qubit 0 being the least significant bit.
            target: Index of the target qubit, which is flipped wherever the control is 1.
            state_vector: The state vector to act on. It is not modified.

        Returns:
            A new state vector of the same shape, with the CNOT applied.
        """
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
    def measurement_all(state_vector: StateVector, number_of_shots: int) -> dict[str, int]:
        """Sample the state ``number_of_shots`` times and tally the outcomes.

        The return value has the same structure as Aer's ``result.get_counts()``: keys are
        zero-padded bitstrings with qubit 0 as the RIGHTMOST character (Qiskit's
        little-endian convention, which is also the convention used by
        :meth:`single_qubit_gate` and :meth:`apply_cnot`), values are how often that
        outcome was drawn. Outcomes that never occurred are left out, exactly as Aer does.

        Args:
            state_vector: The state to sample from. Must be normalised.
            number_of_shots: Number of measurement repetitions to draw.

        Returns:
            A mapping from observed bitstring to the number of times it was drawn. The
            values sum to ``number_of_shots``.

        Raises:
            ValueError: If the state vector is not normalised, i.e. the amplitudes'
                squared moduli do not sum to 1. An explicit raise rather than ``assert``:
                asserts are stripped by ``python -O``, and then the renormalisation below
                would silently rescale an unphysical state instead of reporting that the
                gates lost norm.
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
    def simulation_func(qc: qiskit.QuantumCircuit, number_of_shots: int) -> SimulationResult:
        """Simulate a circuit end to end and return its state vector and counts.

        The circuit is first stripped of Aer's ``save_*`` markers and transpiled into the
        ``u``/``cx`` basis, then its instructions are applied one by one to a register
        initialised in :math:`|0\\dots0\\rangle`. The walk stops at the first ``measure``
        instruction, at which point the state is sampled ``number_of_shots`` times.

        Unsupported instructions are reported on stdout and skipped rather than raising, so
        a circuit containing one silently yields a wrong state.

        Args:
            qc: The circuit to simulate. It is copied before modification, so the caller's
                circuit is left untouched.
            number_of_shots: Number of measurement repetitions, used only if the circuit
                contains a ``measure`` instruction.

        Returns:
            A two-element list ``[state_vector, counts]``. ``state_vector`` is the state at
            the point the walk stopped. ``counts`` is the measurement tally in the format
            of :meth:`measurement_all`, or ``None`` if the circuit contained no ``measure``
            instruction.
        """
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




class simulate_no_einsum:
    """Run a circuit on the own simulator, using an index loop for single-qubit gates.

    Identical to :class:`simulate` except that single-qubit gates are applied by
    :func:`apply_U` rather than by an einsum contraction. Both back ends produce the same
    state; this one exists so the two approaches can be compared.

    Attributes:
        circuit: The circuit to simulate.
        number_of_shots: Number of measurement repetitions.
        return_statevector: Whether :meth:`perform_sim` returns the state vector rather
            than the measurement counts.
    """

    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int, return_statevector: bool) -> None:
        """Store the simulation parameters.

        Args:
            circuit: The circuit to simulate. Any ``save_*`` instruction it carries is
                stripped before running.
            number_of_shots: Number of measurement repetitions.
            return_statevector: If ``True``, :meth:`perform_sim` returns the final state
                vector; if ``False``, it returns the measurement counts.
        """
        self.circuit = circuit
        self.number_of_shots = number_of_shots
        self.return_statevector = return_statevector

    def perform_sim(self) -> StateVector | dict[str, int] | None:
        """Run the circuit on the own simulator and return the requested result.

        Returns:
            The final state vector if ``return_statevector`` is ``True``. Otherwise the
            measurement counts, or ``None`` if the circuit contained no ``measure``
            instruction and therefore produced no counts.
        """
        result = own_simulator_no_einsum.simulation_func(self.circuit, self.number_of_shots)
        if self.return_statevector:
            return result[0]
        return result[1]


class own_simulator_no_einsum:
    """State-vector simulator applying single-qubit gates with an explicit index loop.

    The loop-based sibling of :class:`own_simulator`: :meth:`simulation_func` dispatches
    ``u`` instructions to :func:`apply_U` instead of to an einsum contraction. Everything
    else -- the transpilation, the CNOT handling and the measurement sampling -- is the
    same. Use :class:`simulate_no_einsum` for the usual front end.

    Attributes:
        circuit: The circuit passed at construction.
        number_of_shots: The shot count passed at construction.
    """

    def __init__(self, circuit: qiskit.QuantumCircuit, number_of_shots: int) -> None:
        """Store the circuit and shot count.

        Note:
            Instantiating this class is not required: :meth:`simulation_func` is a static
            method that takes the circuit and the shot count as arguments, and neither
            attribute set here is ever read.

        Args:
            circuit: The circuit to simulate.
            number_of_shots: Number of measurement repetitions.
        """
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
    def single_qubit_gate(
        gate: npt.NDArray[np.complex128],
        qubit_index: int,
        N: int,
        state_vector: StateVector,
    ) -> StateVector:
        """Apply a single-qubit operator to one qubit, via an einsum contraction.

        Kept identical to :meth:`own_simulator.single_qubit_gate`. Note that
        :meth:`simulation_func` of this class does not call it -- it uses :func:`apply_U`
        instead, which is the whole point of the ``no_einsum`` variant.

        Args:
            gate: The 2x2 unitary to apply, as a complex array of shape ``(2, 2)``.
            qubit_index: Index of the qubit the gate acts on, qubit 0 being the least
                significant bit.
            N: Total number of qubits in the register.
            state_vector: The state vector to act on. It is not modified.

        Returns:
            A new state vector of the same shape, with ``gate`` applied to ``qubit_index``.
        """
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
    def apply_cnot(controll: int, target: int, state_vector: StateVector) -> StateVector:
        """Apply a controlled-NOT gate to the state vector.

        Kept identical to :meth:`own_simulator.apply_cnot`.

        Args:
            controll: Index of the control qubit, qubit 0 being the least significant bit.
            target: Index of the target qubit, which is flipped wherever the control is 1.
            state_vector: The state vector to act on. It is not modified.

        Returns:
            A new state vector of the same shape, with the CNOT applied.
        """
        N = len(state_vector)
        copy_state = np.copy(state_vector)
        to_flip = [] #states that have a one 
        to_flip_0 = []#control list, so we do not flip a second time 
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
    def measurement_all(state_vector: StateVector, number_of_shots: int) -> dict[str, int]:
        """Sample the state ``number_of_shots`` times and tally the outcomes.

        Kept identical to :meth:`own_simulator.measurement_all`. The return value has the
        same structure as Aer's ``result.get_counts()``: keys are zero-padded bitstrings
        with qubit 0 as the RIGHTMOST character (Qiskit's little-endian convention, which
        is also the convention used by :meth:`single_qubit_gate` and :meth:`apply_cnot`),
        values are how often that outcome was drawn. Outcomes that never occurred are left
        out, exactly as Aer does.

        Args:
            state_vector: The state to sample from. Must be normalised.
            number_of_shots: Number of measurement repetitions to draw.

        Returns:
            A mapping from observed bitstring to the number of times it was drawn. The
            values sum to ``number_of_shots``.

        Raises:
            ValueError: If the state vector is not normalised, i.e. the amplitudes'
                squared moduli do not sum to 1.
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
    def simulation_func(qc: qiskit.QuantumCircuit, number_of_shots: int) -> SimulationResult:
        """Simulate a circuit end to end and return its state vector and counts.

        Identical to :meth:`own_simulator.simulation_func` except that ``u`` instructions
        are applied with :func:`apply_U` rather than with an einsum contraction.

        Args:
            qc: The circuit to simulate. It is copied before modification, so the caller's
                circuit is left untouched.
            number_of_shots: Number of measurement repetitions, used only if the circuit
                contains a ``measure`` instruction.

        Returns:
            A two-element list ``[state_vector, counts]``. ``state_vector`` is the state at
            the point the walk stopped. ``counts`` is the measurement tally in the format
            of :meth:`measurement_all`, or ``None`` if the circuit contained no ``measure``
            instruction.
        """
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
                state = apply_U(matrix, qubit_indices[0], qc.num_qubits, state)
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
