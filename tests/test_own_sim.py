import block_quantenschaltung as qs



def test_qc():
    test_circ = qs.qiskit.QuantumCircuit(2)
    test_circ.h(0)
    test_circ.cx(0, 1)
    test_circ.save_statevector()

    aer_test = qs.mock_simulate(test_circ, 1000, True)
    own_test = qs.simulate(test_circ, 1000, True)

    aer_res = aer_test.perform_sim()
    own_res = own_test.perform_sim()

    assert aer_res == own_res