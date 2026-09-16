import block_quantenschaltung as qs
import qiskit
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

qc = qiskit.QuantumCircuit(2)

qc.h(0)
qc.measure_all()

Test = qs.simulate(qc, 1000)

result = Test.perform_sim()

print(result.get_counts(qc))