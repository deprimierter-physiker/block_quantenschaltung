import block_quantenschaltung as qs

circ = qs.qiskit.QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
circ.save_statevector()
circ.measure_all()

test = qs.mock_simulate(circ, 1000, False)
res = test.perform_sim()

print(res)