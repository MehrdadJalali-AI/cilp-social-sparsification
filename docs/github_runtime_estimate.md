# GitHub runtime estimate (Stage A diagnostic)

**Diagnostic setting:** seed 0, removal rate 0.5 only, CPU, frozen protocol.

## Observed diagnostic wall-clock

| Method | train_seconds (logged) | Notes |
|---|---|---|
| random + dspar + ER + ILP-GCN + NeuralSparse + PTDNet | ~123 s wall for all six @ one budget | Cheap methods |
| Task-only (`cailp_a31`) | **518.0 s** | teacher_n=40 |
| CILP (`cailp_multi`) | **590.9 s** | teacher_n=40 |

## Diagnostic checks (seed 0 @ 0.5)

- Dataset dims: 37,700 nodes / 289,003 undirected edges / feat 4005 / 2 classes — OK
- Split reuse: `github_seed0.pt` — OK
- Exact budget: retained_edge_ratio = 0.49999827 (= `m - round(0.5 m)` / m) — OK
- Finite Macro-F1 / structural metrics — OK
- Device: CPU (MPS unused) — OK
- Serialization: valid JSON list — OK
- No duplicate keys in diagnostic files — OK

## Extrapolation (protocol unchanged)

| Scope | Estimate |
|---|---|
| One CILP seed (all 9 budgets; train once) | ~10–15 min (dominated by teacher; ~591 s train) |
| One Task-only seed | ~9–12 min (~518 s train) |
| One complete 7-method scientific seed (incl. Resistance proxy once after dedupe; runner still executes dspar+ER) | ~25–40 min |
| Full 10-seed grid | **~4–7 hours** wall-clock on this CPU host |

**Protocol not modified based on estimate.**

## Next step

Move incomplete 1-budget diagnostic JSON files aside, then launch resumable full grid:

```bash
python scripts/run_full_grid.py --datasets github --seeds 0 1 2 3 4 5 6 7 8 9
```
