# Coding conventions

## Naming

### Code
We follow the follwing naming conventions
- Classes: CamelCase
- Functions: camelCase
- Variables/Arguments: snake_case

### Test Functions
- testnames: testModule_testName
 
## Formatting
We use (black)[https://black.readthedocs.io/en/stable/]

## Imports
Only absolute imports are accepted


# Testing
SaQC comes with an extensive test suite based on [pytest](https://docs.pytest.org/en/latest/). In order to 
run all tests use:
```sh
python -m pytest .
```

