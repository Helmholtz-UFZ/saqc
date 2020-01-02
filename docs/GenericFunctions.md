# Generic Functions
Generic Functions provide a possibility to implement simple quality checks
directly within the configuration using a simple, Python based extension 
language.

## Specification
Generic funtions are used in the same manner as their
[non-generic counterparts](docs/FunctionDescriptions.md). The basic 
signature looks like that:
```sh
generic(func=<expression>, flag=<flagging_constant>)
```
where `<expression>` is composed of the [supported constructs](#supported-constructs)
and `<flag_constant>` is either one of the predefined
[flagging constants](docs/ParameterDescriptions.md#flagging-constants) or any value supported
by the flagger in use.

## Examples

### Simple comparisons

#### Task
Flag all values of variable `x` when variable `y` falls below a certain threashold

#### Configuration file

| varname | test                  |
|---------|-----------------------|
| `x`     | `generic(func=y < 0)` |

### Calculations

#### Task
Flag all values of variable `x` that exceed 3 standard deviations of variable `y`

#### Configuration file

| varname | test                              |
|---------|-----------------------------------|
| `x`     | `generic(func=this > std(y) * 3)` |

### Special functions

#### Task
Flag variable `x` where variable `y` is flagged and variable `x` has missing values

#### Configuration file

| varname | test                                              |
|---------|---------------------------------------------------|
| `x`     | `generic(func=this > isflagged(y) & ismissing(z)` |


## Variable References
All variables of the processed dataset are available within generic functions, so 
arbitrary cross references are possible. The variable of intereset 
is furthermore available with the special reference `this`, so the second 
[example](#calculations) could be rewritten as: 

| varname | test                           |
|---------|--------------------------------|
| `x`     | `generic(func=x > std(y) * 3)` |

When referencing other variables, their flags will be respected during evaluation
of the generic expression. So, in the example above only previously
unflagged values of `x` and `y` are used within the expression `x > std(y)*3`. 


## Supported constructs

### Operators

#### Comparsions

The following comparison operators are available:
| Operator | Description                                                                                        |
|----------|----------------------------------------------------------------------------------------------------|
| `==`     | `True` if the values of the operands are equal                                                     |
| `!=`     | `True` if the values of the operands are not equal                                                 |
| `>`      | `True` if the values of the left operand are greater than the values of the right operand          |
| `<`      | `True` if the values of the left operand are smaller than the values of the right operand          |
| `>=`     | `True` if the values of the left operand are greater or equal than the values of the right operand |
| `<=`     | `True` if the values of the left operand are smaller or equal than the values of the right operand |

#### Arithmetics
The following arithmetic operators are supported:
| Operator | Description    |
|----------|----------------|
| `+`      | addition       |
| `-`      | substraction   |
| `*`      | multiplication |
| `/`      | division       |
| `**`     | exponantion    |
| `%`      | modulus        |

#### Bitwise
The bitwise operators also act as logical operators in comparison chains 
| Operator | Description       |
|----------|-------------------|
| `&`      | binary and        |
| &vert;   | binary or         |
| `^`      | binary xor        |
| `~`      | binary complement |

### Functions

All functions expect a [variable reference](#variable-references)
as the only non-keyword argument (see [here](#special-functions))

| Name        | Description                       |
|-------------|-----------------------------------|
| `abs`       | absolute values of a variable     |
| `max`       | maximum value of a variable       |
| `min`       | minimum value of a variable       |
| `mean`      | mean value of a variable          |
| `sum`       | sum of a variable                 |
| `std`       | standard deviation of a variable  |
| `len`       | the number of values for variable |
| `ismissing` | check for missing values          |
| `isflagged` | check for flags                   |

### Constants
Generic functions support the same constants as normal functions, a detailed 
list is available [here](docs/ParameterDescriptions.md#constants).
