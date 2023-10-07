# Activate env
./Scripts/Activate.ps1

# Install requirements
poetry install

# Linter checks
pre-commit install
pre-commit run -a

# Create new directories
New-Item -ItemType Directory dataset -ErrorAction Ignore
New-Item -ItemType Directory saved_model -ErrorAction Ignore
