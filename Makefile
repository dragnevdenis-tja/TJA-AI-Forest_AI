# Makefile for Forest Audio AI

.PHONY: generate-data train evaluate test docker-build docker-up clean

generate-data:
	PYTHONPATH=. python backend/pipelines/data_generation/run_generation.py --n_samples 10000

train:
	PYTHONPATH=. python backend/ml/structured_model/train.py

evaluate:
	PYTHONPATH=. python backend/evaluation/audio_eval.py
	PYTHONPATH=. python backend/evaluation/risk_eval.py
	PYTHONPATH=. python backend/evaluation/ablation.py
	PYTHONPATH=. python backend/evaluation/leakage_check.py

test:
	PYTHONPATH=. python -m pytest tests/

docker-build:
	docker build -t forest-audio-api .

docker-up:
	docker-compose up --build

clean:
	rm -rf __pycache__
	rm -rf backend/**/__pycache__
	rm -rf tests/__pycache__
