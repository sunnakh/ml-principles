# ML from First Principles

A learning and engineering project implementing foundational machine learning
algorithms in Python and NumPy, supported by mathematical explanations,
independent tests, reference comparisons, and failure analysis.

**Status:** the repository foundation, linear-regression contract and derivation,
StandardScaler, regression metrics, and training-mean baseline are implemented and
tested. Linear-regression implementation, reference comparison, experiments, and
benchmark results remain incomplete. This is an educational library, not production
software.

## Quality checks

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src
uv run --locked pytest
uv build
```

The tests cover package imports, StandardScaler behavior, regression metrics,
input contracts, nonmutation, and relevant numerical failures. CI runs formatting,
linting, typing, fast tests, and a package build. Expensive or external checks should
use `slow` or `integration` markers.

## First milestone

StandardScaler, full MSE, MAE, R², and a training-mean baseline now support the
next step: linear regression with batch gradient descent. Validate the estimator
with hand-computable cases, numerical gradient checks, and a trusted reference.
Experiment with feature scaling, learning rate, outliers, and correlated features
before moving to another model.

### Algorithm progress

| Component | Status | Evidence or next step |
|---|---|---|
| Linear-regression contract and derivation | Documented | [Problem framing, full-MSE derivation, API contract, stopping behavior, and failure analysis](docs/algorithms/linear_regression.md) |
| StandardScaler | Implemented and tested | [Dense float64 implementation](src/ml_principles/preprocessing.py) with [contract and edge-case tests](tests/test_preprocessing.py) |
| MSE, MAE, R², and mean baseline | Implemented and tested | [Validated metric implementation](src/ml_principles/metrics.py) with [hand-computed, contract, and numerical tests](tests/test_metrics.py) |
| Batch-gradient-descent linear regression | Next | Implement the documented fit/predict, state, convergence, and failure contracts |
| Linear-regression tests and reference comparison | Planned | Add hand-worked, gradient, invariant, and least-squares checks |
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
