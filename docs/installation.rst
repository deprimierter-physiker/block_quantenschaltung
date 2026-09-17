.. _installation:

Installation
============

``block_quantenschaltung`` is a small quantum-circuit simulator built on top of
:mod:`numpy` and :mod:`qiskit`. This page describes how to get it running, either as a
dependency of your own project or as a checkout you intend to develop on.

Requirements
------------

* **Python 3.14 or newer** (the version pinned in ``.python-version``).
* A C-capable platform supported by NumPy and Qiskit Aer -- no compiler is needed by this
  package itself, but the dependencies ship as binary wheels for Linux, macOS and Windows.

The runtime dependencies are declared in ``pyproject.toml`` and are installed automatically:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``numpy>=2.5.3``
     - State-vector storage and all linear algebra
   * - ``qiskit>=2.5.2``
     - Circuit construction and transpilation to the ``u``/``cx`` basis
   * - ``qiskit-aer>=0.17.2``
     - Reference simulator the own implementation is validated against
   * - ``matplotlib>=3.11.2``
     - Plotting of measurement results
   * - ``pytest>=9.1.1``
     - Test runner
   * - ``pre-commit>=4.6.2``
     - Git hooks for formatting checks

.. _installation-uv:

Installing with uv (recommended)
--------------------------------

The project is built with the `uv <https://docs.astral.sh/uv/>`_ build backend
(``uv_build``), so ``uv`` is the path of least resistance. If you do not have it yet:

.. code-block:: console

   $ curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux / macOS

.. code-block:: doscon

   > powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Clone the repository and let ``uv`` create the virtual environment and resolve everything in
one step:

.. code-block:: console

   $ git clone git@github.com:deprimierter-physiker/block_quantenschaltung.git
   $ cd block_quantenschaltung
   $ uv sync

``uv sync`` creates ``.venv/`` next to ``pyproject.toml``, installs the runtime dependencies
**and** the ``dev`` dependency group (Sphinx, needed to build these docs), and installs
``block_quantenschaltung`` itself in editable mode. If you only want the runtime
dependencies, pass ``--no-dev``:

.. code-block:: console

   $ uv sync --no-dev

Prefix commands with ``uv run`` to execute them inside that environment without activating
it manually:

.. code-block:: console

   $ uv run python -c "import block_quantenschaltung"

Alternatively, activate the environment in the usual way:

.. code-block:: console

   $ source .venv/bin/activate        # Linux / macOS

.. code-block:: doscon

   > .venv\Scripts\activate           # Windows

Installing with pip
-------------------

If you would rather not use ``uv``, a standard virtual environment works as well. The
``uv_build`` backend is fetched automatically by pip as a build requirement, so no extra
setup is needed:

.. code-block:: console

   $ git clone https://github.com/deprimierter-physiker/block_quantenschaltung.git
   $ cd block_quantenschaltung
   $ python3.14 -m venv .venv
   $ source .venv/bin/activate
   $ pip install .

For development, install it in editable mode together with Sphinx:

.. code-block:: console

   $ pip install -e .
   $ pip install "sphinx>=9.1.0"

.. note::

   The ``dev`` dependency group lives in ``[dependency-groups]``, which pip cannot read.
   Install the development tools explicitly, as shown above, or use :ref:`uv
   <installation-uv>`, which understands dependency groups natively.

Setting up the git hooks
------------------------

The repository ships a ``pre-commit`` configuration (trailing-whitespace, end-of-file
fixer, YAML and large-file checks). Enable it once per clone:

.. code-block:: console

   $ uv run pre-commit install

From then on the hooks run on every ``git commit``. To check the whole tree on demand:

.. code-block:: console

   $ uv run pre-commit run --all-files

Verifying the installation
--------------------------

Simulate a two-qubit Bell state and compare the own simulator against Qiskit Aer:

.. code-block:: python

   import block_quantenschaltung as qs

   circuit = qs.qiskit.QuantumCircuit(2)
   circuit.h(0)
   circuit.cx(0, 1)
   circuit.save_statevector()

   own = qs.simulate_no_einsum(circuit, 1000, True).perform_sim()
   aer = qs.mock_simulate(circuit, 1000, True).perform_sim()

   print(own)   # [0.707+0j, 0j, 0j, 0.707+0j]
   print(aer)

Or simply run the test suite, which performs the same comparison:

.. code-block:: console

   $ uv run pytest

All tests should pass. A ``ModuleNotFoundError`` for ``block_quantenschaltung`` means the
package was not installed into the active environment -- re-run ``uv sync`` or
``pip install -e .``.

Building the documentation
--------------------------

With the ``dev`` group installed, build this documentation from the ``docs/`` directory:

.. code-block:: console

   $ cd docs
   $ uv run make html

.. code-block:: doscon

   > cd docs
   > uv run make.bat html

The rendered HTML lands in ``docs/_build/html/index.html``.

Troubleshooting
---------------

``requires-python`` mismatch
   ``pip`` or ``uv`` refuses to install because the interpreter is older than 3.14. Install
   a matching interpreter -- ``uv python install 3.14`` will fetch one for you, and
   ``uv sync`` then picks it up automatically via ``.python-version``.

``TranspilerError`` when simulating
   Aer's ``save_*`` instructions cannot be transpiled into the ``u``/``cx`` basis. Attach
   ``save_statevector()`` *after* transpiling the circuit, not before.

Qiskit Aer wheel not found
   ``qiskit-aer`` does not publish wheels for every platform/Python combination. Check the
   available wheels on PyPI, and fall back to the pure Qiskit path if none matches your
   platform -- only :class:`mock_simulate` needs Aer; the own simulator does not.
