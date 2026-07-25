"""Single-objective (CF-GNNExplainer-style) counterfactual teacher for RQ11 / A31."""

from __future__ import annotations

from src.counterfactual.exact_teacher import CFCoefficients, ExactCounterfactualTeacher


class SingleObjectiveTeacher(ExactCounterfactualTeacher):
    def __init__(self, coefficients: CFCoefficients | None = None) -> None:
        super().__init__(coefficients=coefficients, single_objective=True)
