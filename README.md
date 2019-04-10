## Dependencies
- numpy
- pandas
- numba
- pyyaml

## Test Syntax
### Specification
- Test specifications are written in [YAML](https://en.wikipedia.org/wiki/YAML, "Wikipedia") and contain:
  + A test name, either on of the pre-defined tests or 'generic'
  + Optionally a set of parametes. These should be given in
    json-object or yaml/python-dictionary style (i.e. {key: value})
  + test name and parameter object/dictionary need to be seperated by comma
- Example: `limits, {min: 0, max: 100}`
#### Optional Test Parameters
- `flag`:
  The value to set (more precisely the value to pass to the flagging component) if the tests
  does not pass
- `flag_period`:
  + if a value is flagged, so is the given time period following the timestamp of that value
  + Number followed by a frequency specification, e.g. '5min', '6D'.
    A comprehensive list of the supported frequies can be found in the table 'Offset Aliases' in the [Pandas Docs](http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects "Pandas Docs"). The (probably) most common options are also listed below:

    | frequency string | description |
    |------------------|-------------|
    | `D`              | one day     |
    | `H`              | one hour    |
    | `T` or `min`     | one minute  |
    | `S`              | one second  |
- `flag_values`:
  + Number
  + if a value is flagged, so are the next n previously unflagged values
- `assign_to`:
  + String
  + Assign the test result to a new columns given as a value to assign
### Predefined Tests

| name  | required parameters | optional parameters | description                             |
|-------|---------------------|---------------------|-----------------------------------------|
| `mad` | `z`, `length`       | `deriv = 1`         | mean absolute deviation with measure of <br> central tendency `z` and an <br> rolling window of size `length`. Optionally <br> `deriv`'s derivate of  the dataset is <br> calculated first.                       |
|       |                     |                     |                                         |
    

### User Defined Test
User defined tests allow to specify simple quality checks directly within the configuration.
#### Specification
- Test name: `generic`
- The parameter 'func' followed by an expression needs to be given
- Example: generic, `{func: (thisvar > 0) & ismissing(othervar)}`
#### Restrictions
- only the operators and functions listed below are available
- all checks need to be conditional expression and have to return an array of boolean values, 
  all other expressions are rejected. This limitation is enforced to somewhat narrow the 
  scope of the system and therefore the potential to mess things up and might as well be 
  removed in the future.
#### Syntax
- standard Python syntax
- all variables within the configuration file can be used
#### Supported Operators
- all arithmetic operators
- all comparison operators
- bitwise operators: and, or, xor, complement (`&`, `|`, `^`, `~`)
#### Supported functions

| function name | description                                                      |
|---------------|------------------------------------------------------------------|
| `abs`         | absolute values of a variable                                    |
| `max`         | maximum value of a variable                                      |
| `min`         | minimum value of a variable                                      |
| `mean`        | mean value of a variable                                         |
| `sum`         | sum of a variable                                                |
| `std`         | standard deviation of a variable                                 |
| `len`         | the number of values of variable                                 |
| `ismissing`   | check for missing values (nan and a possibly user defined value) |

#### Referencing Semantics
If another variable is reference within an generic test, the flags from that variable are
propagated to the checked variable.

For example:
Let `var1` and `var2` be two variables of a given dataset and `func: var1 > mean(var1)` 
the condition wheter to flag `var2`. The result of the check can be described
as `isflagged(var1) & istrue(func())`.
