
.PHONY: help ensure-uv install notebook-kernel test test-all lint format clean

# uv installs to ~/.local/bin, which is often missing from make's PATH.
export PATH := $(HOME)/.local/bin:$(PATH)

help:
	@echo "Available commands:"
	@echo "  make install          Install all dependencies"
	@echo "  make notebook-kernel  Register .venv for Cursor/Jupyter notebooks"
	@echo "  make test             Run fast tests"
	@echo "  make test-all         Run all tests including slow/integration"
	@echo "  make lint             Run linters"
	@echo "  make format           Auto-format code"
	@echo "  make clean            Remove build artifacts and caches"

ensure-uv:
	@command -v uv >/dev/null 2>&1 || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)

install: | ensure-uv
	uv sync --all-extras --compile-bytecode
	uv pip install --python .venv/bin/python -e ./third_party/afterburner

notebook-kernel: | ensure-uv
	uv pip install --python .venv/bin/python ipykernel
	.venv/bin/python -m ipykernel install --user --name obfuscation-atlas --display-name "Obfuscation Atlas"
	mkdir -p /workspace/.jupyter
	.venv/bin/python -m ipykernel install --prefix=/workspace/.jupyter --name obfuscation-atlas --display-name "Obfuscation Atlas"
	ln -sfn /workspace/obfuscation-atlas/.venv /workspace/.venv
	@mkdir -p /workspace/.vscode
	@python -c "\
	import json, pathlib; \
	p = pathlib.Path('/workspace/.vscode/settings.json'); \
	s = json.loads(p.read_text()) if p.exists() else {}; \
	s['python.defaultInterpreterPath'] = '/workspace/obfuscation-atlas/.venv/bin/python'; \
	s['python-env.workspaceSearchPaths'] = ['/workspace/obfuscation-atlas']; \
	s['jupyter.interactiveWindow.creationMode'] = 'perFile'; \
	s['jupyter.interactiveWindow.textEditor.executeSelection'] = True; \
	p.write_text(json.dumps(s, indent=4) + '\n')"
	@echo "Wrote .vscode/settings.json with interpreter path"

test:
	uv run pytest tests/ -v

test-all:
	uv run pytest tests/ -v -m ""

lint:
	uv run ruff check .

format:
	uv run ruff check . --fix
