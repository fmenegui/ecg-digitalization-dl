.PHONY: test lint data cache train-cpu
test:
	pytest -q
lint:
	ruff check ecgdl tests scripts benchmarks
data:
	python scripts/fetch_ptbxl.py --out data/ptbxl --train 1500 --val 100 --test 100
cache:
	ecgdl cache pages --bank data/ptbxl/bank.npz --split train --out data/cache/train --n 600 --synth configs/synth.yaml
	ecgdl cache pages --bank data/ptbxl/bank.npz --split val --out data/cache/val --n 24 --seed 101 --synth configs/synth.yaml
	ecgdl cache crops --pages data/cache/train --out data/cache/train_crops --config configs/crop_cpu.yaml --seed 1
	ecgdl cache crops --pages data/cache/val --out data/cache/val_crops --config configs/crop_cpu.yaml --seed 2
train-cpu:
	ecgdl train configs/trace_cpu.yaml
	ecgdl train configs/page_cpu.yaml
