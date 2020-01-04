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
| min       | float     |               | upper bound for valid values |
| max       | float     |               | lower bound for valid values |


The function flags all values outside the closed interval
$`[`$`min`, `max`$`]`$.

## seasonalRange

```
sesonalRange(min, max, startmonth=1, endmonth=12, startday=1, endday=31)
```

| parameter  | data type   | default value | description                  |
| ---------  | ----------- | ----          | -----------                  |
| min        | float       |               | upper bound for valid values |
| max        | float       |               | lower bound for valid values |
| startmonth | integer     | `1`           | interval start month         |
| endmonth   | integer     | `12`          | interval end month           |
| startday   | integer     | `1`           | interval start day           |
| endday     | integer     | `31`          | interval end day             |

The function does the same as `range` but only, if the timestamp of the
data-point lies in a time interval defined by day and month only. 
The year is **not** used by the interval calculation. The left interval
boundary is defined by `startmonth` and `startday`, the right by `endmonth`
and `endday`. Both boundaries are inclusive. If the left side occurs later
in the year than the right side, the interval is extended over the change of
year (e.g. an interval of [01/12, 01/03], will flag values in December,
January and February).

NOTE: Only works for time-series-like datasets.


## isolated

```
isolated(window, group_size=1, continuation_range='1min') 

```

| parameter    | data type                                                     | default value | description                                                            |
|--------------|---------------------------------------------------------------|---------------|------------------------------------------------------------------------|
| group_window | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | Maximum size of an isolated group, see condition (1).                  |
| gap_window   | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | Minimum size of the gap separating isolated, see condition (2) and (3) |

The function flags arbitrary large groups of values, if they are surrounded by sufficiently
large data gaps. A gap is defined as group of missing and/or flagged values.

A continuous group of values
$`x_{k}, x_{k+1},...,x_{k+n}`$ with timestamps $`t_{k}, t_{k+1}, ..., t_{k+n}`$
is considered to be isolated, if:
1. $` t_{k+n} - t_{k} \le `$ `group_window`
2. None of the values $` x_i, ..., x_{k-1} `$, with $`t_{k-1} - t_{i} \ge `$ `gap_window` is valid and unflagged
3. None of the values $` x_{k+n+1}, ..., x_{j} `$, with $`t_{j} - t_{k+n+1} \ge `$ `gap_window` is valid and unflagged


## missing

```
missing(nodata=NaN)
```

| parameter | data type  | default value  | description |
| --------- | ---------- | -------------- | ----------- |
| nodata    | any        | `NAN`          | Value associated with missing data |


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
| flag      | float/[flagging constant](docs/ParameterDescriptions.md#flagging-constants) | GOOD          | flag to force |

The functions sets the given flag, ignoring previous flag values.

