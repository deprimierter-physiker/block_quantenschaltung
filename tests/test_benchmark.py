"""Benchmarks comparing the three simulator back ends against each other.

Three implementations of the same job are measured on identical circuits:

* **qiskit** -- :class:`block_quantenschaltung.mock_simulate`, i.e. Qiskit Aer's
  compiled C++ ``statevector`` method. The reference both for speed and correctness.
* **einsum** -- :class:`block_quantenschaltung.simulate`, which applies single-qubit
  gates with :func:`numpy.einsum` and CNOTs with
  :meth:`block_quantenschaltung.own_simulator.apply_cnot`.
* **numba** -- :class:`block_quantenschaltung.simulate_no_einsum`, which applies
  single-qubit gates with :func:`block_quantenschaltung.apply_U` and CNOTs with
  :func:`block_quantenschaltung.apply_CNOT_clean`, both compiled by ``numba.njit``.

Run them with::

    uv run pytest tests/test_benchmark.py --benchmark-group-by=param:qubits

Ordinary test runs can skip the timings with ``--benchmark-skip``; ``--benchmark-only``
runs nothing else.

Note:
    ``numba`` compiles a function on its first call, which costs a few seconds and would
    otherwise be charged to whichever benchmark ran first. The :func:`warm_up_jit`
    fixture pays that cost once, before any measurement starts.
"""

import numpy as np
import pytest
import qiskit

import block_quantenschaltung as qs

#: Register sizes to benchmark. The einsum back end applies CNOTs with a quadratic
#: algorithm, so it grows steeply; 12 qubits already takes seconds per repetition.
QUBIT_COUNTS = [4, 8, 12]

#: Repetitions per register size, kept low for the large ones so the suite stays quick.
ROUNDS = {4: 20, 8: 10, 12: 3}

#: Gate depth of the random test circuits, before transpilation into the u/cx basis.
DEPTH = 10


@pytest.fixture(scope="session", autouse=True)
def warm_up_jit() -> None:
    """Trigger numba's one-off compilation before any benchmark is timed.

    Both JIT-compiled functions are called once on a tiny state. Without this the first
    benchmark to touch them would absorb roughly 2.6 s of compilation and report the
    numba back end as far slower than it is.
    """
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0
    qs.apply_U(np.eye(2, dtype=complex), 0, 2, state)
    qs.apply_CNOT_clean(0, 1, state)


def build_circuit(qubits: int) -> qiskit.QuantumCircuit:
    """Build the random circuit that every back end is measured on.

    The circuit is transpiled into the ``u``/``cx`` basis up front so that the
    benchmarks measure simulation rather than transpilation, and so that all three back
    ends see exactly the same instruction sequence. ``save_statevector`` is attached
    afterwards, since it is an Aer marker that cannot itself be transpiled; the own
    simulators strip it again before running.

    Args:
        qubits: Number of qubits in the register.

    Returns:
        A transpiled circuit carrying a ``save_statevector`` instruction.
    """
    circuit = qiskit.circuit.random.random_circuit(qubits, DEPTH, measure=False, seed=7)
    circuit = qiskit.transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    circuit.save_statevector()
    return circuit


@pytest.fixture(params=QUBIT_COUNTS, ids=lambda n: f"{n}q")
def qubits(request: pytest.FixtureRequest) -> int:
    """Parametrise the benchmarks over :data:`QUBIT_COUNTS`.

    Args:
        request: The pytest fixture request carrying the current parameter.

    Returns:
        The number of qubits for this run.
    """
    return request.param


@pytest.fixture
def circuit(qubits: int) -> qiskit.QuantumCircuit:
    """Provide the benchmark circuit for the current register size.

    Args:
        qubits: Number of qubits, injected by the :func:`qubits` fixture.

    Returns:
        The transpiled circuit from :func:`build_circuit`.
    """
    return build_circuit(qubits)


def statevector_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    """Compare two state vectors up to a global phase.

    A global phase is physically unobservable, and the back ends do not agree on one, so
    a plain :func:`numpy.allclose` would report false mismatches.

    Args:
        a: First state vector.
        b: Second state vector.
        atol: Absolute tolerance passed through to :func:`numpy.allclose`.

    Returns:
        ``True`` if the two vectors describe the same state.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    phase = np.vdot(a, b)
    if abs(phase) < 1e-12:
        return False
    phase = phase / abs(phase)
    return bool(np.allclose(a, b * np.conj(phase), atol=atol, rtol=1e-5))


@pytest.mark.parametrize("qubits", QUBIT_COUNTS, ids=lambda n: f"{n}q")
def test_backends_agree(qubits: int) -> None:
    """All three back ends must produce the same state before timing them.

    A benchmark of implementations that disagree would be meaningless, so this guards
    the comparison itself.

    Args:
        qubits: Number of qubits, supplied by :func:`pytest.mark.parametrize`.
    """
    circuit = build_circuit(qubits)
    aer = qs.mock_simulate(circuit, 1, True).perform_sim()
    einsum = qs.simulate(circuit, 1, True).perform_sim()
    numba = qs.simulate_no_einsum(circuit, 1, True).perform_sim()

    assert statevector_close(aer, einsum), f"einsum disagrees with Aer at {qubits} qubits"
    assert statevector_close(aer, numba), f"numba disagrees with Aer at {qubits} qubits"


def test_qiskit(benchmark, circuit: qiskit.QuantumCircuit, qubits: int) -> None:
    """Benchmark Qiskit Aer's compiled state-vector simulator.

    Args:
        benchmark: The ``pytest-benchmark`` fixture.
        circuit: The circuit to simulate, from the :func:`circuit` fixture.
        qubits: Register size, used to pick the repetition count.
    """
    simulator = qs.mock_simulate(circuit, 1, True)
    benchmark.pedantic(simulator.perform_sim, rounds=ROUNDS[qubits], iterations=1, warmup_rounds=1)


def test_einsum(benchmark, circuit: qiskit.QuantumCircuit, qubits: int) -> None:
    """Benchmark the own simulator using :func:`numpy.einsum` for single-qubit gates.

    Args:
        benchmark: The ``pytest-benchmark`` fixture.
        circuit: The circuit to simulate, from the :func:`circuit` fixture.
        qubits: Register size, used to pick the repetition count.
    """
    simulator = qs.simulate(circuit, 1, True)
    benchmark.pedantic(simulator.perform_sim, rounds=ROUNDS[qubits], iterations=1, warmup_rounds=1)


def test_numba(benchmark, circuit: qiskit.QuantumCircuit, qubits: int) -> None:
    """Benchmark the own simulator using the ``numba``-compiled gate kernels.

    Args:
        benchmark: The ``pytest-benchmark`` fixture.
        circuit: The circuit to simulate, from the :func:`circuit` fixture.
        qubits: Register size, used to pick the repetition count.
    """
    simulator = qs.simulate_no_einsum(circuit, 1, True)
    benchmark.pedantic(simulator.perform_sim, rounds=ROUNDS[qubits], iterations=1, warmup_rounds=1)
