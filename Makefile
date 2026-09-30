.PHONY: install test quick run

install:
	python3 -m venv --system-site-packages .venv
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/python -m unittest tests.test_metrics -v

quick:
	.venv/bin/python -m src.run_experiment --quick

run:
	.venv/bin/python -m src.run_experiment
