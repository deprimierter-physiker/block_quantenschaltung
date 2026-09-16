import block_quantenschaltung as qs

circ = qs.qiskit.QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
circ.measure_all()

test = qs.simulate(circ, 1000)
res = test.perform_sim()

print(res.get_counts(circ))