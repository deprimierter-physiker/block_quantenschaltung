import qiskit  # type: ignore[import-untyped]

import block_quantenschaltung as qs

circ = qs.qiskit.QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
circ.save_statevector()
circ.measure_all()

test = qs.mock_simulate(circ, 1000, False)
res = test.perform_sim()

print(res)
qc = qiskit.QuantumCircuit(2)

qc.h(0)
qc.measure_all()

Test = qs.mock_simulate(qc, 1000, False)

result = Test.perform_sim()

print(result.get_counts(qc))
