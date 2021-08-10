# Color Macros
RED = `tput setaf 1`
GREEN = `tput setaf 2`
YELLOW = `tput setaf 3`
BLUE = `tput setaf 4`
MAGENTA = `tput setaf 5`
CYAN = `tput setaf 6`
NC = `tput sgr0`

help:
# instructions on how to use make file.
	@echo "$(MAGENTA)make install$(NC) for fresh install"
	@echo "$(BLUE)make conda-update$(NC) to sync conda environment"
	@echo "$(YELLOW)make pip-packages$(NC) to install pip packages"

# Macros for checking anaconda installation and environment.
REQUIRED_ENV = HandwritingBCI

install:
	@make conda-update
	@make install-pip

install-pip:
	@make pip-tools
	@make pip-packages

conda-update:
	conda env update --prune -f environment.yml
	echo "!!!! Run Right now:\nconda activate $(REQUIRED_ENV)"

pip-tools:
	pip install pip-tools

pip-packages:
	pip-compile -v requirements/prod.in && pip-compile -v requirements/dev.in
	pip-sync -v requirements/prod.txt requirements/dev.txt