import block_quantenschaltung as qs




def statevector_close(a, b, atol=1e-8):
    a = qs.np.asarray(a)
    b = qs.np.asarray(b)

    if a.shape != b.shape:
        return False

    a = a / qs.np.linalg.norm(a)
    b = b / qs.np.linalg.norm(b)

    phase = qs.np.vdot(a, b)
    if abs(phase) < 1e-12:
        return False

    phase = phase / abs(phase)
    return qs.np.allclose(a, b * qs.np.conj(phase), atol=atol, rtol=1e-5)

def test_qc():
    test_circ = qs.qiskit.QuantumCircuit(2)
    test_circ.h(0)
    test_circ.cx(0, 1)
    test_circ.save_statevector()

    aer_test = qs.mock_simulate(test_circ, 1000, True)
    own_test = qs.simulate(test_circ, 1000, True)

    aer_res = aer_test.perform_sim()
    own_res = own_test.perform_sim()

    assert statevector_close(aer_res, own_res)  # fails today: TypeError in simulate.perform_sim

# MISSING TEST: nothing exercises the counts path (return_statevector=False). That is the
# path where own_simulator returns a list[int] of basis indices while Aer returns a counts
# dict, so the mismatch documented in measurement_all() is currently untested.
# MISSING TEST: no circuit with more than 3 qubits, and no direct unit test of
# apply_cnot / single_qubit_gate against a reference operator.

def test_qc_random():
    # Order matters: save_statevector() must be attached AFTER transpiling. It is an Aer
    # marker, not a gate, so transpile() cannot translate it into the u/cx basis and raises
    # TranspilerError at every optimization level if it is already in the circuit.
    test_circ = qs.qiskit.circuit.random.random_circuit(3, 5, measure=False, seed=0)
    test_circ = qs.qiskit.transpile(test_circ, basis_gates = ["u", "cx"], optimization_level=0)
    test_circ.save_statevector()

    own_test = qs.simulate(test_circ, 1000, True)
    aer_test = qs.mock_simulate(test_circ, 1000, True)
    aer_res = aer_test.perform_sim()
    own_res = own_test.perform_sim()

    assert statevector_close(aer_res, own_res)

def test_counts_match_aer():
    """The counts path: own_simulator must return the same structure as Aer's get_counts()."""
    test_circ = qs.qiskit.QuantumCircuit(2)
    test_circ.h(0)
    test_circ.cx(0, 1)
    test_circ.measure_all()

    shots = 1000
    aer_res = qs.mock_simulate(test_circ, shots, False).perform_sim()

    qs.np.random.seed(0)
    own_res = qs.simulate(test_circ, shots, False).perform_sim()

    assert isinstance(own_res, dict)
    assert sum(own_res.values()) == shots
    # A Bell state can only give '00' or '11', with probability 1/2 each.
    assert set(own_res) <= {"00", "11"}
    assert set(own_res) == set(aer_res)
    for bitstring, count in own_res.items():
        assert abs(count / shots - aer_res[bitstring] / shots) < 0.1
