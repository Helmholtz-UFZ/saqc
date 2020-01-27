# Miscellaneous

A collection of unrelated quality check functions.

## Index

- [range](#range)
- [seasonalRange](#seasonalrange)
- [isolated](#isolated)
- [missing](#missing)
- [clear](#clear)
- [force](#force)


## range

```
range(min, max)
```
| parameter | data type | default value | description                  |
| --------- | --------- | ------------- | -----------                  |
| min       | float     |               | The upper bound for valid values |
| max       | float     |               | The lower bound for valid values |


The function flags all values outside the closed interval
$`[`$`min`, `max`$`]`$.

## seasonalRange

```
sesonalRange(min, max, startmonth=1, endmonth=12, startday=1, endday=31)
```

| parameter  | data type   | default value | description                  |
| ---------  | ----------- | ----          | -----------                  |
| min        | float       |               | The upper bound for valid values |
| max        | float       |               | The lower bound for valid values |
| startmonth | integer     | `1`           | The interval start month         |
| endmonth   | integer     | `12`          | The interval end month           |
| startday   | integer     | `1`           | The interval start day           |
| endday     | integer     | `31`          | The interval end day             |

The function does the same as `range`, but only if the timestamp of the
data-point lies in a defined interval, which is build from days and months only. 
In particular, the *year* is not considered in the Interval. 

The left 
boundary is defined by `startmonth` and `startday`, the right boundary by `endmonth`
and `endday`. Both boundaries are inclusive. If the left side occurs later
in the year than the right side, the interval is extended over the change of
year (e.g. an interval of [01/12, 01/03], will flag values in December,
January and February).

NOTE: Only works for time-series-like datasets.


## isolated

```
isolated(window, gap_window, group_window) 

```

| parameter    | data type                                                     | default value | description                                                            |
|--------------|---------------------------------------------------------------|---------------|------------------------------------------------------------------------|
| gap_window   | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | The minimum size of the gap before and after a group of valid values, which makes this group considered as isolated. See condition (2) and (3) |
| group_window | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | The maximum size of an isolated group of valid data. See condition (1).                  |

The function flags arbitrary large groups of values, if they are surrounded by sufficiently
large data gaps. A gap is defined as group of missing and/or flagged values.

A continuous group of values
$`x_{k}, x_{k+1},...,x_{k+n}`$ with timestamps $`t_{k}, t_{k+1}, ..., t_{k+n}`$
is considered to be isolated, if:
1. $` t_{k+n} - t_{k} \le `$ `group_window`
2. None of the values $` x_i, ..., x_{k-1} `$, with $`t_{k-1} - t_{i} \ge `$ `gap_window` is valid or unflagged
3. None of the values $` x_{k+n+1}, ..., x_{j} `$, with $`t_{j} - t_{k+n+1} \ge `$ `gap_window` is valid or unflagged


## missing

```
missing(nodata=NaN)
```

| parameter | data type  | default value  | description |
| --------- | ---------- | -------------- | ----------- |
| nodata    | any        | `NAN`          | A value that defines missing data |


The function flags all values indicating missing data.

## clear

```
clear()
```

The funcion removes all previously set flags.

## force

```
force(flag)
```
| parameter | data type                | default value | description   |
| --------- | -----------              | ----          | -----------   |
| flag      | float/[flagging constant](docs/ParameterDescriptions.md#flagging-constants) | GOOD     | The flag that is set unconditionally |

The functions overwrites all previous set flags with the given flag.

