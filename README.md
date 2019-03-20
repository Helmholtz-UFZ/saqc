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

### Not Sure
Parameters given within a test specification generally target on of the following components:
- test function
- flagger
- the general flagging operation (i.e. extensions of flags to an given temporal period)
The question how to solve this differentiation needs to be answered:
- The easier option (at least from a user/usage standpoint) is to simply throw everthing into on dictionary,
  and pass the entire thing to all the relevant functions/methods.
- The more complex option is to enforce a sperate dictionary for every collection of related parameters. This
  would allow to target the internal parameter passing mor specifically

### User Defined Test
#### Syntax
- standard Python syntax
- all variables within the configuration file can be used
