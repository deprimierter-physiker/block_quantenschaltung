import block_quantenschaltung as qs
import qiskit
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

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

Test = qs.simulate(qc, 1000)

result = Test.perform_sim()

print(result.get_counts(qc))
