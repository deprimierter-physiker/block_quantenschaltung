import block_quantenschaltung as qs

# BUG (test): both tests below compare with a bare `==`. `aer_res` is a qiskit Statevector
# (or a counts dict) while `own_res` would be a raw numpy array (or a single bitstring), so
# the comparison either raises "truth value of an array is ambiguous" or can never hold.
# Even for two correct statevectors, exact equality fails on floating-point rounding, and
# two states differing only by a global phase are physically identical but compare unequal.
# FIX: np.allclose(...) up to global phase for statevectors; a tolerance-based comparison
# (shot noise ~ sqrt(shots)) for counts, not exact dict equality.

#circ = qs.qiskit.circuit.random.random_circuit(2, 2, measure=True)

def test_qc():
    test_circ = qs.qiskit.QuantumCircuit(2)
    test_circ.h(0)
    test_circ.cx(0, 1)
    test_circ.save_statevector()

    aer_test = qs.mock_simulate(test_circ, 1000, True)
    own_test = qs.simulate(test_circ, 1000, True)

    aer_res = aer_test.perform_sim()
    own_res = own_test.perform_sim()

    assert aer_res == own_res  # BUG: see note at top of file

def test_qc_random():
    # NOTE: random_circuit(..., measure=True) plus save_statevector() means the statevector
    # is saved AFTER the measurements, so the Aer reference is a collapsed (random) state.
    # Comparing that against a deterministic simulator run is not a well-posed test - the
    # two need not agree even when both are correct. Seed the circuit and drop the measure,
    # or compare counts instead of the statevector.
    test_circ = qs.qiskit.circuit.random.random_circuit(3, 5, measure=True)
    test_circ.save_statevector()

    aer_test = qs.mock_simulate(test_circ, 1000, True)
    own_test = qs.simulate(test_circ, 1000, True)

    aer_res = aer_test.perform_sim()
    own_res = own_test.perform_sim()

    assert aer_res == own_res