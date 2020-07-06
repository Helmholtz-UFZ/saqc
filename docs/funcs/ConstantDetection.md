# Constant Detection

A collection of quality check routines to find constant values and plateaus.

## Index

- [constants_flagBasic](#constants_flagbasic)
- [constants_flagVarianceBased](#constants_flagvariancebased)


## constants_flagBasic

```
constants_flagBasic(window, thresh=0)
```

| parameter | data type                                                             | default value | description                                                                                                                  |
|-----------|-----------------------------------------------------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------|
| window    | integer/[offset string](docs/ParameterDescriptions.md#offset-strings) |               | The minimum count or duration in which the values must be constant to be considered as plateau candidates. See condition (1) |
| thresh    | float                                                                 |             0 | The maximum difference between values to be still considered as constant. See condition (2)                                     |

This functions flags plateaus/series of constant values of length `window` if
their difference is smaller than `thresh`.

A set of consecutive values $`x_n, ..., x_{n+k}`$ of a time series $`x_t`$
is considered to be constant, if:
1. $`k \ge `$ `window`
2. $`|x_n - x_{n+s}| \le `$ `thresh`, $`s \in {1,2, ..., k}`$


## constants_flagVarianceBased

```
constants_flagVarianceBased(window="12h", thresh=0.0005,
                            max_missing=None, max_consec_missing=None)
```

| parameter          | data type                                                     | default value | description                                                                                            |
|--------------------|---------------------------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------|
| window             | [offset string](docs/ParameterDescriptions.md#offset-strings) | `12h`         | The minimum duration in which the values must be constant to be considered as plateau candidates. See condition (1) |
| thresh             | float                                                         | `0.0005`      | The maximum variance of a group of values, to still consider them as constant. See condition (2)               |
| max_missing        | integer                                                       | `None`        | The maximum count of missing values that are allowed in the `window`. If not set, this condition is ignored and infinity missing values are allowed.|
| max_consec_missing | integer                                                       | `None`        | The maximum count of *consecutive* missing values, that are allowed in the `window`. If not set, this condition is ignored and infinity consecutive missing values are allowed. |


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
