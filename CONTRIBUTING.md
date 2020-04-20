# Development Environment
We recommend an virtual python environment for development. The setup process consists of the follwing simply steps:

1. Create a fresh environment with:
   ```sh
   python -m venv saqc_dev
   ```
2. Activate the created environment
   ```
   source saqc_dev/bin/activate
   ```
3. Install the dependencies
   ```sh
   python -m pip install -r requirements.txt
   ```
 
# Testing
SaQC comes with an extensive test suite based on [pytest](https://docs.pytest.org/en/latest/).
In order to run all tests execute:
```sh
python -m pytest .
```

# Coding conventions

## Naming

### Code
We follow the follwing naming conventions
- Classes: CamelCase
- Functions: camelCase
- Variables/Arguments: snake_case

### Test Functions
- testnames: [testmodule_]flagTestName
 
## Formatting
We use (black)[https://black.readthedocs.io/en/stable/] with a line length if 120 characters.
Within the `SaQC` root directory run `black -l 120`

## Imports
Only absolute imports are accepted


