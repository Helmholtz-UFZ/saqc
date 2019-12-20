# Adding new functions/tests
## Interface
- All user exposed quality checks and processing algorithms need to implement the following interface:
  ```python
  der yourTestFunction(
      data: pandas.DataFrame,
      field: str,
      flagger: saqc.flagger.BaseFlagger,
      *args: Any,
      **kwargs: Any
      ) -> (pd.DataFrame, saqc.flagger.BaseFlagger)
  ```
## Argument Descriptions

| Name      | Description                                                                                      |
| ----      | -----------                                                                                      |
| `data`    | The actual dataset                                                                               |
| `field`   | The field/column within `data`, the function is checking/processing                              |
| `flagger` | A instance of a flagger, responsible for the translation of test results into quality attributes |
| `args`    | Any other arguments needed to parametrize the function                                           |
| `kwargs`  | Any other keyword arguments needed tp parametrize the function                                   |

## Integrate into SaQC
In order make your function available to the system it needs to be registered. We provide the decorator 
[`register`](saqc/functions/register.py) in the module `saqc.functions.register`, to integrate your 
test functions into SaQC and expose them via a name of your choice. A complete, yet useless example might
look like that:

```python
@register("myFunc")
def yourTestFunction(data, field, flagger, *args, **kwargs):
    return data, flagger
```

## Example
The function [`flagRange`](saqc/funcs/functions.py) provides a simple, yet complete implementation of the
entire test function contribution process. You might want to look into this implementation before you start.


# Adding a new flagger
TODO

# Testing
SaQC comes with an extensive test suite based on [pytest](https://docs.pytest.org/en/latest/). In order to 
run all tests use:
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
- testnames: testModule_testName
 
## Formatting
We use (black)[https://black.readthedocs.io/en/stable/]

## Imports
Only absolute imports are accepted
