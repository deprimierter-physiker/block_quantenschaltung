"""Benchmark the three simulator back ends on whole random circuits.

Where ``plot_gate_benchmark.py`` times the individual gate kernels, this script runs a
complete circuit through each back end and asks the practical question: which one would
you actually use?

* **Qiskit Aer** -- :class:`block_quantenschaltung.mock_simulate`, the compiled C++
  reference.
* **einsum** -- :class:`block_quantenschaltung.simulate`, which applies single-qubit
  gates with :func:`numpy.einsum` and CNOTs with the original quadratic
  :meth:`own_simulator.apply_cnot`.
* **numba** -- :class:`block_quantenschaltung.simulate_no_einsum`, which uses the
  JIT-compiled :func:`apply_U` and :func:`apply_CNOT_clean`.

Two panels: absolute time per circuit, and the slowdown of the own back ends relative
to Aer.

Colours, theme and label placement are shared with ``plot_gate_benchmark.py`` so the two
figures read as one set. Importing that module also pins every numerical backend to a
single thread, which is why it is imported before NumPy.

Run it with::

    uv run python benchmarks/plot_circuit_benchmark.py
    uv run python benchmarks/plot_circuit_benchmark.py --replot   # reuse cached timings

It writes ``circuit_benchmark_light.png`` and ``circuit_benchmark_dark.png`` next to
itself, alongside ``circuit_benchmark.json``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Sets OMP_NUM_THREADS and friends to 1 before NumPy is imported anywhere.
from plot_gate_benchmark import PALETTE, THEME, measure, place_labels  # noqa: E402

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import qiskit  # noqa: E402

import block_quantenschaltung as qs  # noqa: E402

#: Register sizes. Aer and the numba back end are measured across the whole range.
QUBITS = [4, 6, 8, 10, 12, 14, 16]

#: The einsum back end applies CNOTs with a quadratic algorithm, so it is cut off early:
#: at 14 qubits a single circuit already takes the better part of a minute.
QUBITS_EINSUM = [n for n in QUBITS if n <= 12]

#: Gate depth of the random circuits, before transpilation into the u/cx basis.
DEPTH = 10

#: Seed for :func:`qiskit.circuit.random.random_circuit`, so the circuits are fixed.
SEED = 7

#: Series label -> (front-end class, register sizes, palette slot).
BACKENDS = {
    "Qiskit Aer": (qs.mock_simulate, QUBITS, "loop"),
    "einsum": (qs.simulate, QUBITS_EINSUM, "numpy"),
    "numba": (qs.simulate_no_einsum, QUBITS, "numba"),
}


def build_circuit(qubits: int) -> qiskit.QuantumCircuit:
    """Build the random circuit that every back end is measured on.

    The circuit is transpiled into the ``u``/``cx`` basis up front, so the benchmark
    measures simulation rather than transpilation and every back end sees the same
    instruction sequence. ``save_statevector`` is attached afterwards because it is an
    Aer marker that cannot itself be transpiled; the own simulators strip it again.

    Args:
        qubits: Number of qubits in the register.

    Returns:
        A transpiled circuit carrying a ``save_statevector`` instruction.
    """
    circuit = qiskit.circuit.random.random_circuit(qubits, DEPTH, measure=False, seed=SEED)
    circuit = qiskit.transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    circuit.save_statevector()
    return circuit


def states_agree(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare two state vectors up to an unobservable global phase.

    Args:
        first: One state vector.
        second: The other state vector.

    Returns:
        ``True`` if both describe the same physical state.
    """
    a = np.asarray(first) / np.linalg.norm(first)
    b = np.asarray(second) / np.linalg.norm(second)
    phase = np.vdot(a, b)
    if abs(phase) < 1e-12:
        return False
    return bool(np.allclose(a, b * np.conj(phase / abs(phase)), atol=1e-8, rtol=1e-5))


def collect() -> dict[str, tuple[list[int], list[float]]]:
    """Time every back end over the register sizes, checking they agree first.

    Returns:
        A mapping from back-end label to ``(qubit counts, milliseconds per circuit)``.
    """
    series: dict[str, tuple[list[int], list[float]]] = {label: ([], []) for label in BACKENDS}

    for n in QUBITS:
        circuit = build_circuit(n)
        instructions = len(circuit.data)
        reference = None

        for label, (front_end, sizes, _) in BACKENDS.items():
            if n not in sizes:
                continue

            simulator = front_end(circuit, 1, True)
            state = np.asarray(simulator.perform_sim())
            if reference is None:
                reference = state
            elif not states_agree(reference, state):
                raise AssertionError(f"{label} disagrees with the reference at {n} qubits")

            elapsed = measure(lambda: simulator.perform_sim())
            series[label][0].append(n)
            series[label][1].append(elapsed)
            print(f"  {label:12} {n:2d} qubits  {instructions:4d} instructions  "
                  f"{elapsed:10.3f} ms", flush=True)

    return series


def draw(series: dict[str, tuple[list[int], list[float]]], mode: str, destination: Path) -> None:
    """Render the two-panel figure for one colour mode and write it to disk.

    Args:
        series: Timings as returned by :func:`collect`.
        mode: Either ``"light"`` or ``"dark"``.
        destination: Path of the PNG to write.
    """
    colours, ink = PALETTE[mode], THEME[mode]
    aer_sizes, aer_times = series["Qiskit Aer"]
    aer_by_size = dict(zip(aer_sizes, aer_times))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.0), dpi=200)
    figure.patch.set_facecolor(ink["surface"])

    for axis in axes:
        axis.set_facecolor(ink["surface"])
        axis.set_yscale("log")
        axis.grid(True, which="major", color=ink["grid"], linewidth=0.8, zorder=0)
        axis.grid(True, which="minor", color=ink["grid"], linewidth=0.4, alpha=0.6, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(ink["grid"])
        axis.tick_params(colors=ink["secondary"], labelsize=9)
        axis.set_xlabel("Qubits", color=ink["secondary"], fontsize=10)
        axis.set_xticks(QUBITS)
        axis.set_xlim(min(QUBITS) - 0.6, max(QUBITS) + 0.6)

    # Left panel: absolute cost of simulating one circuit.
    for label, (_, _, slot) in BACKENDS.items():
        sizes, times = series[label]
        axes[0].plot(sizes, times, color=colours[slot], linewidth=2, marker="o", markersize=5,
                     markeredgecolor=ink["surface"], markeredgewidth=1.2, zorder=3)
    axes[0].set_title("Time per circuit", color=ink["primary"], fontsize=12, pad=10, loc="left")
    axes[0].set_ylabel("Milliseconds (best of N)", color=ink["secondary"], fontsize=10)

    # Right panel: how many times slower than Aer. Aer itself is the reference line,
    # not a series - a flat line at 1 would carry no information.
    axes[1].axhline(1.0, color=ink["grid"], linewidth=1.4, linestyle="--", zorder=1)
    # Anchored at the right end, where the data has long since climbed away from 1x;
    # at the left end it would sit on top of the numba curve.
    axes[1].annotate("Qiskit Aer = 1x", (QUBITS[-1], 1.0), textcoords="offset points",
                     xytext=(-2, 6), ha="right", fontsize=8.5, color=ink["secondary"])
    for label, (_, _, slot) in BACKENDS.items():
        if label == "Qiskit Aer":
            continue
        sizes, times = series[label]
        ratios = [t / aer_by_size[n] for n, t in zip(sizes, times)]
        axes[1].plot(sizes, ratios, color=colours[slot], linewidth=2, marker="o", markersize=5,
                     markeredgecolor=ink["surface"], markeredgewidth=1.2, zorder=3)
    axes[1].set_title("Slowdown relative to Qiskit Aer", color=ink["primary"], fontsize=12,
                      pad=10, loc="left")
    axes[1].set_ylabel("Factor", color=ink["secondary"], fontsize=10)

    figure.canvas.draw()
    place_labels(axes[0], [(series[label][0][-1], series[label][1][-1], label)
                           for label in BACKENDS], ink["primary"])
    place_labels(axes[1], [(series[label][0][-1],
                            series[label][1][-1] / aer_by_size[series[label][0][-1]], label)
                           for label in BACKENDS if label != "Qiskit Aer"], ink["primary"])

    handles = [plt.Line2D([], [], color=colours[slot], linewidth=2, marker="o", markersize=5,
                          markeredgecolor=ink["surface"], markeredgewidth=1.2, label=label)
               for label, (_, _, slot) in BACKENDS.items()]
    legend = figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
                           frameon=False, ncol=3, fontsize=9, handletextpad=0.5,
                           columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(ink["secondary"])

    figure.suptitle("Full circuit simulation", color=ink["primary"], fontsize=14,
                    x=0.008, ha="left", y=0.975)
    figure.text(0.008, 0.918,
                f"Random circuits, depth {DEPTH}, seed {SEED}, transpiled to u/cx. Log scale, "
                "single-threaded; einsum stops at 12 qubits because its CNOT is quadratic.",
                color=ink["secondary"], fontsize=9, ha="left")

    figure.tight_layout(rect=(0, 0, 0.90, 0.885))
    figure.savefig(destination, dpi=200, facecolor=ink["surface"])
    plt.close(figure)
    print(f"wrote {destination}", flush=True)


def main() -> None:
    """Collect the timings once, cache them, and render both colour modes."""
    here = Path(__file__).resolve().parent
    cache = here / "circuit_benchmark.json"

    if "--replot" in sys.argv and cache.exists():
        raw = json.loads(cache.read_text())
        series = {label: (value[0], value[1]) for label, value in raw.items()}
        print(f"reusing {cache}", flush=True)
    else:
        series = collect()
        cache.write_text(json.dumps({k: [v[0], v[1]] for k, v in series.items()}, indent=2) + "\n")
        print(f"wrote {cache}", flush=True)

    for mode in ("light", "dark"):
        draw(series, mode, here / f"circuit_benchmark_{mode}.png")


if __name__ == "__main__":
    main()
