.. Block Quantenschaltung documentation master file.

Block Quantenschaltung
======================

Dokumentation für das Paket ``block-quantenschaltung``.

.. toctree::
   :maxdepth: 2
   :caption: Inhalte:

   installation
   api
   notebooks/test_jnb


Benchmarks
----------

.. figure:: /images/gate_benchmark_dark.png
   :alt: Runtime comparisons between single qubit and CNOT gates
   :width: 600px
   :align: center

   Abbildung 1: Laufzeit der einzelnen Gatter-Kernel -- Einzelqubit-Gatter und CNOT --
   fuer die drei Backends.

.. figure:: /images/circuit_benchmark_dark.png
   :alt: Runtime of a full circuit simulation per back end and the slowdown relative to Qiskit Aer
   :width: 600px
   :align: center

   Abbildung 2: Laufzeit einer vollstaendigen Schaltungssimulation pro Backend sowie der
   Verlangsamungsfaktor gegenueber Qiskit Aer.
