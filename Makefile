.PHONY: setup cutechess format
.SILENT: setup cutechess format

VENV = .venv
PY_DIRS = chesSLM.py chesslm
PIP = $(VENV)/bin/pip

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

cutechess:
	curl -s https://api.github.com/repos/cutechess/cutechess/releases/latest | grep "browser_download_url" | grep "x86_64.AppImage" | cut -d '"' -f 4 | xargs curl -L -o CuteChess.AppImage && chmod +x CuteChess.AppImage

format:
		echo "  sorting python imports..."
		$(VENV)/bin/ruff check --select I --fix $(PY_DIRS)
		echo "  formatting python code..."
		$(VENV)/bin/ruff format $(PY_DIRS)
