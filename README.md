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
