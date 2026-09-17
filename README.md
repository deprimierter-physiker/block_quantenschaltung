# block_quantenschaltung

This repository is created by Pascal and Marius for the "Blockpraktikum: Simulation von Quantenschaltungen" at the university of Stuttgart.

The repository contains a python library created using uv, which can simulate qiskit quantum circuits. This library is tested via the playground foulder and the tests foulder. Running pytest in the tests foulder can confirm the functionality of the library.
The simulation can be performed via the AerSimulator, an own implementation using np.einsum and an implementation without np.einsum.

Hopefully, one can download the project via
´´´
pip install git+https://github.com/pairinteraction/rydstate
´´´