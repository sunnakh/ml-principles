# Algorithm name

Status: planned / in progress / verified. Link the implementation and tests once present.

## Problem and intuition

State the task, inputs/outputs, tiny worked example, baseline, relevant downstream
decision, and error costs. Label hypothetical use cases and impact explicitly.

## Assumptions and applicability

When does this approach fit? When does it fail? Distinguish assumptions needed for
prediction from assumptions needed for statistical inference.

## Mathematics

Define symbols and shapes. Derive the prediction rule, objective, update/solution,
regularization convention, and convergence criteria. Full MSE is the default.

## API and implementation decisions

Document supported inputs, dtype, state, random seed, validation, unsupported
cases, numerical stability, and allowed NumPy primitives. Explain alternatives.

## Verification

Include hand-computable examples, invariants, gradient checks if applicable, edge
cases, and reference comparisons. Explain tolerances and equivalent solutions.

## Experiments and results

Record data provenance/license/version, split rationale, preprocessing fit scope,
seed, configuration, command, metric/baseline, actual result, and interpretation.
Change one factor at a time. Leave unmeasured results explicitly pending.

## Complexity and benchmarks

Define dimensions and fit/predict/storage complexity. Record hardware/software,
thread count, warmup, repeated-trial statistic/spread, and measurement boundaries.

## Failure analysis

For each investigated failure: symptom, mechanism, controlled experiment, observed
evidence, mitigation, and remaining trade-off. Include useful plots with explanation.

## Operational considerations and limitations

Explain schema/feature order, memory/latency, relevant monitoring and label delays,
and what would be needed before deployment. State what has not been demonstrated.

## Reproduce and references

Provide exact runnable commands and sources supporting the math and data. Mark
completion only after the AGENTS.md gates are satisfied.
