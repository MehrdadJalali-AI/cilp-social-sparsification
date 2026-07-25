"""DSpar-style degree / effective-resistance motivated sparsification."""

from src.sparsification.baselines.classical import effective_resistance_proxy

def dspar_sparsify(data, removal_rate: float):
    return effective_resistance_proxy(data, removal_rate)
