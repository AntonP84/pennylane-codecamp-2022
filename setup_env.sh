# install python3.9
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9
sudo apt install python3.9-dev python3.9-distutils

# install poetry
pip install -U pip setuptools
curl -sSL https://install.python-poetry.org | python3 -

# to install poetry dependencies
poetry env use python3.9
poetry install --sync --no-root

# to add poetry dependencies (unless you already have .toml files)
poetry init
poetry config virtualenvs.in-project true --local
