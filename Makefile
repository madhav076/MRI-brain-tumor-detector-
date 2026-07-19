.PHONY: install train evaluate run test lint format clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

train:
	python scripts/run_training.py

evaluate:
	python scripts/run_evaluation.py

run:
	python scripts/run_app.py

test:
	pytest tests/ --cov=src --cov-report=xml --cov-report=html

lint:
	flake8 .
	mypy src

format:
	black .
	isort .

clean:
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist htmlcov rmdir /s /q htmlcov
	@if exist .mypy_cache rmdir /s /q .mypy_cache
	@if exist .coverage del /f /q .coverage
	@if exist coverage.xml del /f /q coverage.xml
	@for /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo Codebase cleaned.
