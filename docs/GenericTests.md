### User Defined Test
User defined tests allow to specify simple quality checks directly within the
configuration.
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
