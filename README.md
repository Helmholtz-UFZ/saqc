## Dependencies
- numpy
- pandas
- pyyaml

## Test Specification Syntax

### Format yaml
- Pros:
  + a superset of json
  + seems to be more convient than json (no need to quote identifiers)
- Cons:
  + less common than json
  + external dependency

### Specification
A test specification contains:
- A test name, either on of the pre-defined tests or 'generic'
- Optionally a set of parametes. These should be given in
  json-object or yaml/python-dictionary style (i.e. {key: value})
- test name and parameter object/dictionary need to be seperated by comma

### User Defined Test
#### Syntax
- standard Python syntax
- all variables within the configuration file can be used
