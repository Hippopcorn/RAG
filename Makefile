PYTHON = python3
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
MAIN_FILE = src/__main__.py

all: run

run:
	uv run python -m src

install:
	uv sync

debug:
	$(PYTHON) -m pdb $(MAIN_FILE)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf *.pyc
	rm -rf .pytest_cache

lint:
	@echo "--- Running Flake8---"
	python3 -m flake8 . --exclude=.venv 
	@echo "\n--- Running Mypy---"
	python3 -m mypy $(MYPY_FLAGS) .


.PHONY: install run debug clean lint