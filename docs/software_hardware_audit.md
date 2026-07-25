# Software and hardware audit

## Recoverable from repository

| Item | Status |
|---|---|
| Device in experiment logs | `device=cpu` |
| MPS policy | Disabled in `src/utils/io.py::get_device` |
| `environment.yml` | Python 3.10 + conda/pip deps (unpinned torch) |
| `Dockerfile` | `python:3.10-slim` |
| `requirements.txt` | Version floors only |
| `pip freeze` lockfile | **Not present** |
| Exact CPU model for each grid row | **Not stored** |
| Exact OS version for each grid row | **Not stored** (paths suggest macOS user environment) |

## Analysis workstation (table/figure regeneration)

Recorded during local analysis regeneration (not claimed as the unique historical grid host):

- macOS arm64; Python 3.9.6
- PyTorch 2.4.0; PyTorch Geometric 2.6.1; NetworkX 3.2.1; NumPy 1.26.4; SciPy 1.13.1; scikit-learn 1.6.1; matplotlib 3.9.4
- CUDA unavailable; MPS available but unused by protocol

Manuscript SI S27 states these facts transparently.
