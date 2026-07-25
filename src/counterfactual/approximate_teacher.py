"""Alias module for approximate teacher (same API; uses cheaper proxies)."""

from __future__ import annotations

from src.counterfactual.exact_teacher import CFCoefficients, ExactCounterfactualTeacher


class ApproximateCounterfactualTeacher(ExactCounterfactualTeacher):
    """Currently shares ExactTeacher with reduced spectral cost already gated by n.

    Future: replace community/spectral with purely local proxies for very large graphs.
    """

    pass
