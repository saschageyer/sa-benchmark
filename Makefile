VIRTUALENV?=.venv
PORT?=8888

help:
	@echo "Make targets:"
	@echo "  build            build virtual environtment and installing dependencies"
	@echo "  lab              run jupyter lab (default port $(PORT))"

build:
	echo "building virtual environtment and installing dependencies..."
	python -m venv $(VIRTUALENV); \
	source $(VIRTUALENV)/bin/activate; \
	python -m pip install --upgrade pip; \
	python -m pip install -r requirements.txt;

lab:
	source $(VIRTUALENV)/bin/activate; \
	jupyter lab --port=$(PORT)
