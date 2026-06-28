.PHONY: all setup clean run
.SILENT: all setup clean run

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

all:
	@echo "|> no target selected, abort"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

clean:
	rm -rf $(VENV)

run:
	./run.sh
