.PHONY: download audit splits test pilot baselines ablation evaluate tables figures paper

download:
	python scripts/download_data.py --all

audit:
	python scripts/audit_data.py --datasets facebook lastfm github

splits:
	python scripts/build_splits.py --datasets facebook lastfm github --seeds 0 1 2 3 4 5 6 7 8 9

test:
	pytest -q

pilot:
	python scripts/train_cailp.py --dataset lastfm --config configs/experiments/pilot.yaml --seed 0

baselines:
	python scripts/run_baselines.py --datasets lastfm --budgets 0.3 0.5 0.7 --seed 0

ablation:
	python scripts/run_ablation.py --datasets lastfm --seed 0

evaluate:
	python scripts/evaluate_all.py
	python scripts/make_tables.py
	python scripts/make_figures.py

paper:
	python scripts/build_paper.py
