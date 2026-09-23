# ML from First Principles

A learning and engineering project implementing foundational machine learning
algorithms in Python and NumPy, supported by mathematical explanations,
independent tests, reference comparisons, and failure analysis.

**Status:** the repository foundation is complete, and the linear-regression
contract and derivation are documented. Algorithm implementations, algorithm tests,
experiments, and benchmark results are not complete yet. This is an educational
library, not production software.

## Setup

Prerequisite: [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run these commands from the project directory:

```sh
uv sync --locked
uv run --locked python -c "import ml_principles, numpy; print(numpy.__version__)"
```

Python 3.12 is selected by `.python-version`; uv can download it if unavailable.
The local environment is `.venv`. uv runs use it automatically. For a conventional
terminal session on macOS/Linux, activate it with `source .venv/bin/activate`;
leave it with `deactivate`. Select `.venv/bin/python` in your editor.

Optional notebooks:

```sh
uv sync --locked --extra notebooks
uv run --locked --extra notebooks jupyter lab
```

Use the project's Python kernel and keep reusable logic in `src/ml_principles`.
The lockfile includes optional dependencies; the initial installation need not
install them. Normal `uv sync --locked` returns to the default dependency set.

## Quality checks

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src
uv run --locked pytest
uv build
```

The initial test checks packaging only. Add independent algorithm tests as each
implementation is developed. CI runs formatting, linting, typing, fast tests, and
a package build. Expensive/external checks should use `slow`/`integration` markers.

## First milestone

Implement and explain StandardScaler and linear regression with batch gradient
descent, full MSE, MAE, and R². Validate with hand-computable cases, numerical
gradient checks, and a trusted reference. Experiment with feature scaling,
learning rate, outliers, and correlated features before moving to another model.

### Algorithm progress

| Component | Status | Evidence or next step |
|---|---|---|
| Linear-regression contract and derivation | Documented | [Problem framing, full-MSE derivation, API contract, stopping behavior, and failure analysis](docs/algorithms/linear_regression.md) |
| StandardScaler | Next | Implement dense float64 fit/transform behavior and edge cases |
| MSE, MAE, and R² metrics | Planned | Implement after scaling so evaluation uses explicit, tested conventions |
| Batch-gradient-descent linear regression | Planned | Implement after preprocessing and metrics |
| Independent tests and reference comparison | Planned | Add hand-worked, gradient, invariant, and least-squares checks |
| Controlled experiments and benchmarks | Planned | Record measured evidence only after the implementation passes its checks |

The linear-regression document remains **in progress** because implementation,
verification, reference comparison, and experiments are still pending.

## Working conventions

Read [AGENTS.md](AGENTS.md) for the complete working agreement and use the
[algorithm template](docs/templates/algorithm.md) when adding a module.

- `src/ml_principles/`: reusable implementations; NumPy is the runtime dependency.
- `tests/`: independent tests and API contracts.
- `docs/algorithms/`: derivations, experiment evidence, and limitations.
- `examples/`: reproducible examples and thin notebooks.
- `benchmarks/`: measurement scripts and reports with environment details.

scikit-learn is available for development comparisons only. Keep `.venv`, private
data, credentials, and generated binaries out of Git. Commit `uv.lock` alongside
`pyproject.toml`. Add dependencies with `uv add` (runtime), `uv add --dev`
(development), or `uv add --optional notebooks` (notebook tools), then review the
lockfile and run affected checks. Upgrade deliberately rather than during unrelated work.

License: not selected yet. The owner should choose one before inviting public reuse.
