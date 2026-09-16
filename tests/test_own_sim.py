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
    return qs.np.allclose(a, b * phase, atol=atol, rtol=1e-5)

def test_qc():
    test_circ = qs.qiskit.QuantumCircuit(2)
    test_circ.h(0)
    test_circ.cx(0, 1)
    test_circ.save_statevector()

    aer_test = qs.mock_simulate(test_circ, 1000, True)
    own_test = qs.simulate(test_circ, 1000, True)

    aer_res = aer_test.perform_sim()
    own_res = own_test.perform_sim()

    assert statevector_close(aer_res, own_res)  # BUG: see note at top of file

def test_qc_random():
    test_circ = qs.qiskit.circuit.random.random_circuit(3, 5, measure=False)
    test_circ.save_statevector()

    aer_test = qs.mock_simulate(test_circ, 1000, True)
    own_test = qs.simulate(test_circ, 1000, True)

    aer_res = aer_test.perform_sim()
    own_res = own_test.perform_sim()

    assert statevector_close(aer_res, own_res)