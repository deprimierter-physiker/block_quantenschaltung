"""Benchmark the gate kernels and plot how each one scales with the register size.

Two panels, sharing a y-axis so they can be read against each other:

* **Single-qubit gate** -- :meth:`own_simulator.single_qubit_gate` (``numpy.einsum``)
  against :func:`apply_U` (``numba.njit``).
* **CNOT** -- :meth:`own_simulator.apply_cnot` (the original Python loop),
  :func:`apply_CNOT_clean` (``numba.njit``) and :func:`apply_CNOT_reshape`
  (NumPy reshape).

Colour encodes the *approach* rather than the individual function, so the same hue
means the same thing in both panels: blue for the NumPy-vectorised kernel, orange for
the numba-compiled one, aqua for the original Python loop.

Run it with::

    uv run python benchmarks/plot_gate_benchmark.py

It writes ``gate_benchmark_light.png`` and ``gate_benchmark_dark.png`` next to itself.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in CI or a headless shell.

import matplotlib.pyplot as plt
import numpy as np

import block_quantenschaltung as qs

#: Register sizes for the kernels that scale linearly.
QUBITS = list(range(4, 21, 2))

#: The original ``apply_cnot`` is quadratic in the state length, so it is cut off early:
#: at 16 qubits a single call already takes seconds.
QUBITS_QUADRATIC = [n for n in QUBITS if n <= 14]

#: Roughly how long each individual measurement is allowed to take, in seconds.
TIME_BUDGET = 0.25

#: An arbitrary but fixed 2x2 unitary to feed the single-qubit kernels.
GATE = np.array([[0.6, 0.8j], [0.8, -0.6j]], dtype=np.complex128)

#: Categorical slots 1-3 of the validated default palette, light and dark.
PALETTE = {
    "light": {"numpy": "#2a78d6", "numba": "#eb6834", "loop": "#1baf7a"},
    "dark": {"numpy": "#3987e5", "numba": "#d95926", "loop": "#199e70"},
}

#: Chart surface and ink per mode. The dark steps are selected for the dark surface,
#: not an automatic inversion of the light ones.
THEME = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e3e2de"},
    "dark": {"surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7", "grid": "#33322f"},
}


def random_state(num_qubits: int) -> np.ndarray:
    """Build a normalised random state vector.

    Args:
        num_qubits: Size of the register.

    Returns:
        A complex state vector of length ``2**num_qubits``, normalised to 1.
    """
    rng = np.random.default_rng(num_qubits)
    state = rng.normal(size=2**num_qubits) + 1j * rng.normal(size=2**num_qubits)
    return (state / np.linalg.norm(state)).astype(np.complex128)


def measure(function, *args: object) -> float:
    """Time a single call, in milliseconds, as the best of several repetitions.

    The minimum is reported rather than the mean: it is the run least disturbed by
    scheduling noise, which is what "how fast is this kernel" should mean.

    Args:
        function: The callable to time.
        *args: Arguments forwarded to ``function``.

    Returns:
        The fastest observed call time in milliseconds.
    """
    function(*args)  # Warm caches; for a numba kernel this also forces compilation.

    start = time.perf_counter()
    function(*args)
    single = max(time.perf_counter() - start, 1e-9)
    repeats = int(np.clip(TIME_BUDGET / single, 3, 200))

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def collect() -> dict[str, tuple[list[int], list[float]]]:
    """Run every kernel over the register sizes and collect the timings.

    Returns:
        A mapping from series label to ``(qubit counts, milliseconds)``.
    """
    series: dict[str, tuple[list[int], list[float]]] = {}

    def run(label: str, sizes: list[int], call) -> None:
        times = []
        for n in sizes:
            state = random_state(n)
            times.append(measure(call, n, state))
            print(f"  {label:34} {n:2d} qubits  {times[-1]:9.4f} ms", flush=True)
        series[label] = (sizes, times)

    print("Single-qubit gate:", flush=True)
    run("einsum", QUBITS, lambda n, s: qs.own_simulator.single_qubit_gate(GATE, n // 2, n, s))
    run("apply_U (numba)", QUBITS, lambda n, s: qs.apply_U(GATE, n // 2, n, s))

    print("CNOT:", flush=True)
    run("apply_cnot (original)", QUBITS_QUADRATIC, lambda n, s: qs.own_simulator.apply_cnot(0, 1, s))
    run("apply_CNOT_clean (numba)", QUBITS, lambda n, s: qs.apply_CNOT_clean(0, 1, s))
    run("apply_CNOT_reshape", QUBITS, lambda n, s: qs.apply_CNOT_reshape(0, 1, s))

    return series


def place_labels(axis, entries: list[tuple[float, float, str]], colour: str) -> None:
    """Write direct labels at the line ends, nudged apart so they cannot collide.

    Final values can sit arbitrarily close together on a log axis, so the labels are
    laid out in display space: sorted top to bottom, then pushed down whenever a
    neighbour is nearer than the minimum gap.

    Args:
        axis: The axes to annotate.
        entries: One ``(x, y, label)`` per series, in data coordinates.
        colour: Ink colour for the label text.
    """
    minimum_gap = 26.0  # pixels; an 8.5 pt label is ~24 px tall at 200 dpi
    placed = sorted(
        ((x, y, label, axis.transData.transform((x, y))[1]) for x, y, label in entries),
        key=lambda item: -item[3],
    )

    targets: list[float] = []
    for _, _, _, display_y in placed:
        if targets and display_y > targets[-1] - minimum_gap:
            display_y = targets[-1] - minimum_gap
        targets.append(display_y)

    points_per_pixel = 72.0 / axis.figure.dpi
    for (x, y, label, display_y), target in zip(placed, targets):
        axis.annotate(label, (x, y), textcoords="offset points",
                      xytext=(8, (target - display_y) * points_per_pixel),
                      va="center", fontsize=8.5, color=colour, annotation_clip=False)


def draw(series: dict[str, tuple[list[int], list[float]]], mode: str, destination: Path) -> None:
    """Render the two-panel figure for one colour mode and write it to disk.

    Args:
        series: Timings as returned by :func:`collect`.
        mode: Either ``"light"`` or ``"dark"``.
        destination: Path of the PNG to write.
    """
    colours, ink = PALETTE[mode], THEME[mode]

    panels = [
        ("Single-qubit gate", [("einsum", "einsum", "numpy"),
                               ("apply_U (numba)", "apply_U", "numba")]),
        ("CNOT", [("apply_CNOT_reshape", "apply_CNOT_reshape", "numpy"),
                  ("apply_CNOT_clean (numba)", "apply_CNOT_clean", "numba"),
                  ("apply_cnot (original)", "apply_cnot", "loop")]),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True, dpi=200)
    figure.patch.set_facecolor(ink["surface"])

    for axis, (title, members) in zip(axes, panels):
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

        axis.set_title(title, color=ink["primary"], fontsize=12, pad=10, loc="left")
        axis.set_xlabel("Qubits", color=ink["secondary"], fontsize=10)
        axis.set_xticks(QUBITS)
        # Room on the right for the direct labels, which sit outside the data area.
        axis.set_xlim(min(QUBITS) - 0.6, max(QUBITS) + 0.6)

        for key, _, slot in members:
            sizes, times = series[key]
            axis.plot(sizes, times, color=colours[slot], linewidth=2, marker="o",
                      markersize=5, markeredgecolor=ink["surface"], markeredgewidth=1.2,
                      zorder=3)

    # Annotate only once the axes limits are final, so display coordinates are correct.
    figure.canvas.draw()
    for axis, (_, members) in zip(axes, panels):
        # Text carries ink, never the series colour; the coloured mark beside it
        # carries the identity. These labels are also the relief that the light-mode
        # contrast warning requires.
        place_labels(axis, [(series[key][0][-1], series[key][1][-1], short)
                            for key, short, _ in members], ink["primary"])

    axes[0].set_ylabel("Time per gate call (ms, best of N)", color=ink["secondary"], fontsize=10)

    # The legend keys the colours to the three implementation approaches, which is what
    # the hues actually encode; the direct labels name the individual functions.
    handles = [plt.Line2D([], [], color=colours[slot], linewidth=2, marker="o", markersize=5,
                          markeredgecolor=ink["surface"], markeredgewidth=1.2, label=name)
               for slot, name in (("numpy", "NumPy vectorised"),
                                  ("numba", "numba (njit)"),
                                  ("loop", "original Python loop"))]
    legend = figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
                           frameon=False, ncol=3, fontsize=9, handletextpad=0.5, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(ink["secondary"])

    figure.suptitle("Gate kernel scaling", color=ink["primary"], fontsize=14,
                    x=0.008, ha="left", y=0.975)
    figure.text(0.008, 0.918,
                "Lower is better. Log scale; the original CNOT stops at 14 qubits because it is quadratic.",
                color=ink["secondary"], fontsize=9, ha="left")

    # The direct labels live outside the axes, where tight_layout cannot see them,
    # so the right edge is pulled in by hand to keep them inside the canvas.
    figure.tight_layout(rect=(0, 0, 0.87, 0.885))
    figure.savefig(destination, dpi=200, facecolor=ink["surface"])
    plt.close(figure)
    print(f"wrote {destination}", flush=True)


def main() -> None:
    """Collect the timings once, cache them, and render both colour modes.

    Pass ``--replot`` to reuse the cached measurements instead of running the
    benchmark again, which is useful when only the styling changes.
    """
    here = Path(__file__).resolve().parent
    cache = here / "gate_benchmark.json"

    if "--replot" in sys.argv and cache.exists():
        raw = json.loads(cache.read_text())
        series = {label: (value[0], value[1]) for label, value in raw.items()}
        print(f"reusing {cache}", flush=True)
    else:
        series = collect()
        cache.write_text(json.dumps({k: [v[0], v[1]] for k, v in series.items()}, indent=2) + "\n")
        print(f"wrote {cache}", flush=True)

    for mode in ("light", "dark"):
        draw(series, mode, here / f"gate_benchmark_{mode}.png")


if __name__ == "__main__":
    main()
