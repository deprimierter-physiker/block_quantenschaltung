# Blockpraktikum Quantenschaltung

This repository is created by Pascal and Marius for the "Blockpraktikum: Simulation von Quantenschaltungen" at the university of Stuttgart.

The repository contains a python library created using uv, which can simulate qiskit quantum circuits. This library is tested via the playground foulder and the tests foulder. Running pytest in the tests foulder can confirm the functionality of the library.
The simulation can be performed via the AerSimulator, an own implementation using np.einsum and an implementation without np.einsum.
For more information one may read the documentation, created using sphinx.

Hopefully, one can download the project via
```
pip install git+https://github.com/deprimierter-physiker/block_quantenschaltung
```

This work was supported by claude and copilot.

![Runtime comparison between single qubit and CNOT gates.](/images/gate_benchmarkblack.png)
