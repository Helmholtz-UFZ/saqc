## Dependencies
- numpy
- pandas
- pyyaml

## Test Syntax

### Specification
A test specification contains:
- A test name, either on of the pre-defined tests or 'generic'
- Optionally a set of parametes. These should be given in
  json-object or yaml/python-dictionary style (i.e. {key: value})
- test name and parameter object/dictionary need to be seperated by comma

### User Defined Tests
#### Syntax
- standard Python syntax
- all variables within the configuration file can be used
#### Supported Operators
- all arithmetic operators
- all comparison operators
- bitwise operators: and, or, xor, complement (&, |, ^, ~)
#### Supported functions:
- abs: absolute values of a variable
- max: maximum value of a variable
- min: minimum value of a variable
- mean: mean value of a variable
- sum: sum of a variable
- std: standard deviation of a variable
- len: the number of values of variable
- ismissing: check for missing values (nan and a possibly user defined value)
