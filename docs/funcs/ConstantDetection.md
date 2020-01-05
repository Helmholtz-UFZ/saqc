# Constant Detection

A collection of quality check routines to find constant values and plateaus.

## Index

- [constant](#constant)
- [constants_varianceBased](#constants_variancebased)


## constant

```
constant(window, thresh=0)
```

| parameter | data type                                                             | default value | description                                                                                          |
|-----------|-----------------------------------------------------------------------|---------------|------------------------------------------------------------------------------------------------------|
| window    | integer/[offset string](docs/ParameterDescriptions.md#offset-strings) |               | Minimum count or a duration values need to identical to become plateau candidates. See condition (1) |
| thresh    | float                                                                 |             0 | Maximum difference between values to still consider them constant. See condition (2)                 |

This functions flags plateaus/series of constant values of length `window` if
their difference is smaller than `thresh`.

A set of consecutive values $`x_n, ..., x_{n+k}`$ of a time series $`x_t`$
is considered to be constant, if:
1. $`k \ge `$ `window`
2. $`|x_n - x_{n+s}| \le `$ `thresh`, $`s \in {1,2, ..., k}`$


## constants_varianceBased

```
constants_varianceBased(window="12h", thresh=0.0005,
                        max_missing=Inf, max_consec_missing=Inf)
```

| parameter          | data type                                                     | default value | description                                                                                            |
|--------------------|---------------------------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------|
| window             | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | Minimum duration during which values need to identical to become plateau candidates. See condition (1) |
| thresh             | float                                                         | `0.0005`      | Maximum variance of a group of values to still consider them constant. See condition (2)               |
| max_missing        | integer                                                       | `None`        | Maximum number of missing values allowed in `window`, by default this condition is ignored             |
| max_consec_missing | integer                                                       | `None`        | Maximum number of consecutive missing values allowed in `window`, by default this condition is ignored |


This function flags plateaus/series of constant values. Any set of consecutive values
$`x_n,..., x_{n+k}`$ of a timeseries $`x_t`$ is flagged, if:

1. $`k \ge `$`window`
2. $`\sigma(x_n,..., x_{n+k}) \le`$ `thresh`

NOTE:
- Only works for time series
- The time series is expected to be harmonized to an
  [equidistant frequency grid](docs/funcs/TimeSeriesHarmonization.md)
- When `max_missing` or `max_consec_missing` are set, plateaus not 
  fulfilling the respective condition will not be flagged
