# Implemented QC functions

Main documentation of the implemented functions, their purpose and parameters and their description.

## Index

- [Miscellaneous](#miscellaneous)
  - [range](#range)
  - [seasonalRange](#seasonalrange)
  - [isolated](#isolated)
  - [missing](#missing)
  - [clear](#clear)
  - [force](#force)
- [Spike Detection](#spike-detection)
  - [spikes_basic](#spikes_basic)
  - [spikes_simpleMad](#spikes_simplemad)
  - [spikes_slidingZscore](#spikes_slidingzscore)
  - [spikes_spektrumBased](#spikes_spektrumbased)
- [Constant Detection](#constant-detection)
  - [constant](#constant)
  - [constants_varianceBased](#constants_variancebased)
- [Break Detection](#break-detection)
  - [breaks_spektrumBased](#breaks_spektrumbased)
- [Time Series Harmonization](#time-series-harmonization)
  - [harmonize_shift2Grid](#harmonize_shift2grid)
  - [harmonize_aggregate2Grid](#harmonize_aggregate2grid)
  - [harmonize_linear2Grid](#harmonize_linear2grid)
  - [harmonize_interpolate2Grid](#harmonize_interpolate2grid)
  - [harmonize_downsample](#harmonize_downsample)
  - [harmonize](#harmonize)
  - [deharmonize](#deharmonize)
  - [aggregations](#aggregations)
- [Soil Moisture](#soil-moisture)
  - [soilMoisture_plateaus](#soilmoisture_plateaus)
  - [soilMoisture_spikes](#soilmoisture_spikes)
  - [soilMoisture_breaks](#soilmoisture_breaks)
  - [soilMoisture_byFrost](#soilmoisture_byfrost)
  - [soilMoisture_byPrecipitation](#soilmoisture_byprecipitation)
- [Machine Learning](#machine-learning)
  - [machinelearning](#machinelearning)


## Miscellaneous

### range

```
range(min, max)
```
| parameter | data type | default value | description                  |
| --------- | --------- | ------------- | -----------                  |
| min       | float     |               | upper bound for valid values |
| max       | float     |               | lower bound for valid values |


The function flags all values, that exceed the closed interval $`[`$`min`, `max`$`]`$.

### seasonalRange

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

The function does the same as `range` 
(flags all data, that exceed the interval $`[`$`min`, `max`$`]`$), 
but only, if the timestamp of the data-point lies in a time interval defined by day and month only. 
The year is **not** used by the interval calculation. 
The left interval boundary is defined by `startmonth` and `startday`, the right by `endmonth` and `endday`. 
Both boundaries are inclusive. 
If the left side occurs later in the year than the right side, the interval is extended over the change of year 
(e.g. an interval of [01/12, 01/03], will flag values in december, january and february).

Note: Only works for datetime indexed data


### isolated

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


### missing

```
missing(nodata=NaN)
```

| parameter | data type  | default value  | description |
| --------- | ---------- | -------------- | ----------- |
| nodata    | any        | `NAN`          | Value indicating missing values in the passed data. |


The function flags those values in the the passed data series, that are
associated with "missing" data. The missing data indicator (default: `NAN`), can
be altered to any other value by passing this value to the parameter `nodata`.


### clear

```
clear()
```

Remove all previously set flags.

### force

```
force(flag)
```
| parameter | data type                | default value | description   |
| --------- | -----------              | ----          | -----------   |
| flag      | float/GOOD/BAD/UNFLAGGED | GOOD          | flag to force |

Force flags to the given flag value.

## Spike Detection

### spikes_basic
```
spikes_basic(thresh, tolerance, window_size)
```

| parameter   | data type | default value | description                                                                                  |
| ------      | ------    | ------        | ----                                                                                         |
| thresh      | float     |               | Minimum jump margin for spikes. See condition (1).                                           |
| tolerance   | float     |               | Range of area, containing all "valid return values". See condition (2).                       |
| window_size | string    |               | An offset string, denoting the maximum length of "spikish" value courses. See condition (3). |

A basic outlier test, that is designed to work for harmonized, as well as raw
(not-harmonized) data.

The values $`x_{n}, x_{n+1}, .... , x_{n+k} `$ of a passed timeseries $`x`$,
are considered spikes, if:

1. $`|x_{n-1} - x_{n + s}| > `$ `thresh`, $` s \in \{0,1,2,...,k\} `$

2. $`|x_{n-1} - x_{n+k+1}| < `$ `tolerance`

3. $` |y_{n-1} - y_{n+k+1}| < `$ `window_size`, with $`y `$, denoting the series
   of timestamps associated with $`x `$.

By this definition, spikes are values, that, after a jump of margin `thresh`(1),
are keeping that new value level they jumped to, for a timespan smaller than
`window_size` (3), and do then return to the initial value level -
within a tolerance margin of `tolerance` (2).  

Note, that this characterization of a "spike", not only includes one-value
outliers, but also plateau-ish value courses.

### spikes_simpleMad

Flag outlier by simple median absolute deviation test.

```
spikes_simpleMad(winsz="1h", z=3.5)
```

| parameter | data type            | default value | description                                                          |
| --------- | -----------          | ----          | -----------                                                          |
| winsz     | offset-string or int | `"1h"`        | size of the sliding window, where the modified Z-score is applied on |
| z         | float                | `3.5`         | z-parameter the modified Z-score                                     |


The *modified Z-score* [1] is used to detect outlier. 
All values are flagged as outlier, if in any slice of the sliding window, a value fulfills:
```math
 0.6745 * |x - M| > mad * z > 0
```
with $`x, M, mad, z`$: window data, window median, window median absolute deviation, `z`.
The window is moved by one frequency step.

Note: This function should only be applied on normalized data.
 
See also:
[1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

### spikes_slidingZscore

Detect outlier/spikes by a given method in a sliding window.

```
spikes_slidingZscore(winsz="1h", dx="1h", count=1, deg=1, z=3.5, method="modZ")
```

| parameter | data type             | default value | description                                                     |
| --------- | -----------           | ----          | -----------                                                     |
| winsz     | offset-string/integer | `"1h"`        | size of the sliding window, the *method* is applied on          |
| dx        | offset-string/integer | `"1h"`        | the step size the sliding window is continued after calculation |
| count     | integer               | `1`           | the minimal count, a possible outlier needs, to be flagged      |
| deg       | integer               | `1"`          | the degree of the polynomial fit, to calculate the residual     |
| z         | float                 | `3.5`         | z-parameter for the *method* (see description)                  |
| method    | string                | `"modZ"`      | the method outlier are detected with                            |

Parameter notes: 
 - `winsz` and `dx` must be of same type, mixing of offset and integer is not supported and will fail.
 - offset-strings only work with datetime indexed data

The algorithm works as follows:
  1.  a window of size `winsz` is cut from the data
  2.  normalisation - (the data is fit by a polynomial of the given degree `deg`, which is subtracted from the data)
  3.  the outlier detection `method` is applied on the residual, and possible outlier are marked
  4.  the window (on the data) is moved by `dx`
  5.  start over from 1. until the end of data is reached
  6.  all potential outliers, that are detected `count`-many times, are flagged as outlier 

The possible outlier detection methods are *zscore* and *modZ*. 
In the following description, the residual (calculated from a slice by the sliding window) is referred as *data*.

The **zscore** (Z-score) [1] mark every value as possible outlier, which fulfill:
```math
 |r - m| > s * z
```
with $`r, m, s, z`$: data, data mean, data standard deviation, `z`.

The **modZ** (modified Z-score) [1] mark every value as possible outlier, which fulfill:
```math
 0.6745 * |r - M| > mad * z > 0
```
with $` r, M, mad, z `$: data, data median, data median absolute deviation, `z`.

See also:
[1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

### spikes_spektrumBased

```
spikes_spektrumBased(raise_factor=0.15, dev_cont_factor=0.2,
                     noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar",
                     smooth_poly_order=2, filter_window_size=None)
```

| parameter          | data type          | default value | description                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------             | ------             | ------        | ----                                                                                                                                                                                                                                                                                                                                                                                                               |
| raise_factor       | float              | `0.15`        | Minimum margin of value change, a datapoint has to represent, to become a candidate for a spike. See condition (1).                                                                                                                                                                                                                                                                                                                        |
| dev_cont_factor    | float              | `0.2`         | See condition (2).                                                                                                                                                                                                                                                                                                                                                                                                 |
| noise_barrier      | float              | `1`           | Upper bound for noisyness of data surrounding potential spikes. See condition (3).                                                                                                                                                                                                                                                                                                                                 |
| noise_window_range | string             | `"12h"`       | Any offset string. Determines the range of the time window of the "surrounding" data of a potential spike. See condition (3).                                                                                                                                                                                                                                                                                       |
| noise_statistic    | string             | `"CoVar"`     | Operator to calculate noisyness of data, surrounding potential spikes. Either `"Covar"` (=Coefficient od Variation) or `"rvar"` (=relative Variance).                                                                                                                                                                                                                                                               |
| smooth_poly_order  | integer            | `2`           | Order of the polynomial fit, applied with savitsky-Golay-filter.                                                                                                                                                                                                                                                                                                                                                                 |
| filter_window_size | string             | `None`        | Any offset string <br/><br/> Controlls the range of the smoothing window applied with the Savitsky-Golay filter. If `None` (default), the window size will be two times the sampling rate. (Thus, covering 3 values.) If unsure, do not change that value. |


The function detects and flags spikes in input data series by evaluating the
the timeseries' derivatives and applying some conditions to them.

NOTE, that the datast-to-be flagged is supposed to be harmonized to a timeseries with an
equadistant frequency grid.

A datapoint $`x_k`$ of a dataseries $`x`$,
is considered a spike, if:

1. The quotient to its preceeding datapoint exceeds a certain bound:
    * $` |\frac{x_k}{x_{k-1}}| > 1 + `$ `raise_factor`, or
    * $` |\frac{x_k}{x_{k-1}}| < 1 - `$ `raise_factor`
2. The quotient of the datas second derivate $`x''`$, at the preceeding
   and subsequent timestamps is close enough to 1:
    * $` |\frac{x''_{k-1}}{x''_{k+1}} | > 1 - `$ `dev_cont_factor`, and
    * $` |\frac{x''_{k-1}}{x''_{k+1}} | < 1 + `$ `dev_cont_factor`   
3. The dataset, $`X_k`$, surrounding $`x_{k}`$, within `noise_window_range` range,
   but excluding $`x_{k}`$, is not too noisy. Wheras the noisyness gets measured
   by `noise_statistic`:
    * `noise_statistic`$`(X_k) <`$ `noise_barrier`

NOTE, that the derivative is calculated after applying a Savitsky-Golay filter
to $`x`$.

This function is a generalization of the Spectrum based Spike flagging
mechanism as presented in:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
doi:10.2136/vzj2012.0097.

## Constant Detection

### constant

```
constant(window, eps=0)
```

| parameter | data type                                                             | default value | description |
| ------    | ------                                                                | ------        | ----        |
| window    | [offset string](docs/ParameterDescriptions.md#offset-strings)/integer |               |             |
| eps       | float                                                                 | 0             |             |


### constants_varianceBased

```
constants_varianceBased(plateau_window_min="12h", plateau_var_limit=0.0005,
                        var_total_nans=Inf, var_consec_nans=Inf)
```

| parameter          | data type | default value | description |
| ------             | ------    | ------        | ----        |
| plateau_window_min | string    |               | Options <br/> - any offset string <br/> <br/> Minimum barrier for the duration, values have to be continouos to be plateau canditaes. See condition (1). |
| plateau_var_limit  | float     | `0.0005`      | Barrier, the variance of a group of values must not exceed to be flagged a plateau. See condition (2). |
| var_total_nans     | integer   | `Inf`         | Maximum number of nan values allowed, for a calculated variance to be valid. (Default skips the condition.) |
| var_consec_nans    | integer   | `Inf`         | Maximum number of consecutive nan values allowed, for a calculated variance to be valid. (Default skips the condition.) |


Function flags plateaus/series of constant values. Any set of consecutive values
$`x_k,..., x_{k+n}`$ of a timeseries $`x`$ is flagged, if:

1. $`n > `$`plateau_window_min`
2. $`\sigma(x_k,..., x_{k+n})`$ < `plateau_var_limit`

NOTE, that the dataseries-to-be flagged is supposed to be harmonized to an
equadistant frequency grid.

NOTE, that when `var_total_nans` or `var_consec_nans` are set to a value < `Inf`
, plateaus that can not be calculated the variance of, due to missing values,
will never be flagged. (Test not applicable rule.)


## Break Detection

### breaks_spektrumBased

```                            
breaks_spektrumBased(rel_change_min=0.1, abs_change_min=0.01, first_der_factor=10,
                     first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                     scnd_der_ratio_margin_2=10, smooth_poly_order=2,
                     diff_method="raw", filter_window_size="3h")
```

| parameter               | data type | default value | description |
| ------                  | ------    | ------        | ----        |
| rel_change_rate_min     | float     | `0.1`         | Lower bound for the relative difference, a value has to have to its preceeding value, to be a candidate for being break-flagged. See condition (2).|
| abs_change_min          | float     | `0.01`        | Lower bound for the absolute difference, a value has to have to its preceeding value, to be a candidate for being break-flagged. See condition (1).|
| first_der_factor        | float     | `10`          | Factor of the second derivates "arithmetic middle bound". See condition (3).|
| first_der_window_size   | string    | `"12h"`       | Options: <br/> - any offset String <br/> <br/> Determining the size of the window, covering all the values included in the the arithmetic middle calculation of condition (3).|
| scnd_der_ratio_margin_1 | float     | `0.05`        | Range of the area, covering all the values of the second derivatives quotient, that are regarded "sufficiently close to 1" for signifying a break. See condition (5).|
| scnd_der_ratio_margin_2 | float     | `10.0`        | Lower bound for the break succeeding second derivatives quotients. See condition (5). |
| smooth_poly_order       | integer   | `2`           | When calculating derivatives from smoothed timeseries (diff_method="savgol"), this value gives the order of the fitting polynomial calculated in the smoothing process.|
| diff_method             | string    | `"savgol"`     | Options: <br/> - `"savgol"`  <br/> - `"raw"` <br/><br/> Select "raw", to skip smoothing before differenciation. |
| filter_window_size      | Nonetype or string   | `None` | Options: <br/> - `None` <br/> - any offset string <br/><br/> Controlls the range of the smoothing window applied with the Savitsky-Golay filter. If None is passed (default), the window size will be two times the sampling rate. (Thus, covering 3 values.) If you are not very well knowing what you are doing - do not change that value. Broader window sizes caused unexpected results during testing phase.|


The function flags breaks (jumps/drops) in input measurement series by
evaluating its derivatives.

NOTE, that the dataseries-to-be flagged is supposed to be harmonized to an
equadistant frequencie grid.

NOTE, that the derivatives are calculated after applying a savitsky-golay filter
to $`x`$.

A value $`x_k`$ of a data series $`x`$, is flagged a break, if:

1. $`x_k`$ represents a sufficient absolute jump in the course of data values:
    * $`|x_k - x_{k-1}| >`$ `abs_change_min`
2. $`x_k`$ represents a sufficient relative jump in the course of data values:
    * $`|\frac{x_k - x_{k-1}}{x_k}| >`$ `rel_change_min`
3. Let $`X_k`$ be the set of all values that lie within a `first_der_window_range` range around $`x_k`$. Then, for its arithmetic mean $`\bar{X_k}`$, following equation has to hold:
    * $`|x'_k| >`$ `first_der_factor` $` \times \bar{X_k} `$
4. The second derivations quatients are "sufficiently equalling 1":
    * $` 1 -`$ `scnd_der_ratio_margin_1` $`< |\frac{x''_{k-1}}{x_{k''}}| < 1 + `$`scnd_der_ratio_margin_1`
5. The the succeeding second derivatives values quotient has to be sufficiently high:
    * $`|\frac{x''_{k}}{x''_{k+1}}| > `$`scnd_der_ratio_margin_2`

This Function is a generalization of the Spectrum based Spike flagging
mechanism as presented in:

Dorigo,W. et al.: Global Automated Quality Control of In Situ Soil Moisture
Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
doi:10.2136/vzj2012.0097.


## Time Series Harmonization

### harmonize_shift2grid

```
harmonize_shift2Grid(freq, shift_method='nearest_shift', drop_flags=None)
```
| parameter | data type | default value | description |
| --------- | --------- | ------------- | ----------- |
| freq          | string            |                   | Offset string. Detemining the frequency grid, the data shall be shifted to.  |
| shift_method  | string            | `nearest_shift`   | Method, used for shifting of data and flags. See a list of methods below. |
| drop_flags    | list or Nonetype  | `None`              | Flags to be excluded from harmonization. See description of step 3 below. |


The function "harmonizes" the data-to-be-flagged, to match an equidistant 
frequency grid by shifting the datapoints. This is achieved by shifting all data 
points to timestamp values, that are multiples of `freq`.

In detail, the process includes:
 
1. All missing values in the data, identified by `np.nan` 
   get flagged and will be excluded from the shifting process.
   NOTE, that implicitly this step includes a call to `missing` onto the 
   data-to-be-flagged. 
2. Additionally, if a list is passed to `drop_flags`, all the values in data, 
   that are flagged with a flag, listed in `drop_list`, will be excluded from
   shifting - meaning, that they will not affect the further 
   shifting prozess.
3. Depending on the keyword passed to `shift_method`, 
   the data gets shifted, together with its flags,
   to a timestamp that is a multiple of `freq`.
   NOTE, that this step projects the data to an equidistant frequencie grid ranging from the initial to the last timestamp of the data passen and by this,
   will very likely change the size of the dataseries to-be-flagged.
   New sampling in the equidistant freq grid, covering no data in the 
   original dataseries, or only data that got excluded in step (1), 
   will be regarded as representing missing data (Thus get assigned `NaN` value). 
   The original data will be dropped (but can be regained by function 
   `deharmonize`).


`shift_method` keywords::
* `"fshift"`: every grid point gets assigned its ultimately preceeding flag/datapoint 
      if there is one available in the preceeding sampling interval. If not, BAD/np.nan - flag gets assigned.
* `"bshift"`: every grid point gets assigned its first succeeding flag/datapoint 
      if there is one available in the succeeding sampling interval. If not, BAD/np.nan - flag gets assigned.
* `"nearest_shift"`: every grid point gets assigned the closest flag/datapoint in its range. ( range = +/- `freq`/2 ).
   
## harmonize_aggregate2grid

```
harmonize_aggregate2Grid(freq, agg_func, agg_method='nearest_agg', flag_agg_func="max", drop_flags=None)
```
| parameter     | data type         | default value     | description |
| ---------     | ---------         | -------------     | ----------- |
| freq          | string            |                   | Offset string. Determining the sampling rate of the frequency grid, the data shall be aggregated to.  |
| agg_func      | string              |                   | String, signifying a function used for data aggregation. See a table of keywords [here](#aggregations).|
| agg_method    | string            | `nearest_agg`     | Method, determining the range of data and flags aggregation. See a list of methods below. |
| flag_agg_func | string            | `"max"`           |  String, signifying a function used for flags aggregation. See a table of keywords [here](#aggregations).|   
| drop_flags    | list or Nonetype  | `None`              | Flags to be excluded from harmonization. See description of step 2 below. |


The function aggregates the data-to-be-flagged, to match an equidistant 
frequency grid. 
The data aggregagation is carried out, according to the aggregation method `agg_method`, 
the aggregated value is calculated with `agg_func` and gets assigned to a 
timestamp value, that is a multiples of `freq`.

In detail, the process includes:
 
1. All missing values in the data, identified by `np.nan`, 
   get flagged and will be excluded from the aggregation process.
   NOTE, that implicitly this step includes a call to `missing` onto the 
   data-to-be-flagged. 
2. Additionally, if a list is passed to `drop_flags`, all the values in data, 
   that are flagged with a flag, listed in `drop_list`, will be excluded from
   aggregation - meaning, that they will not affect the further 
   aggregation prozess.
3. Depending on the keyword passed to `agg_method`, values get aggregated by 
   `agg_func` and the result, assigned to a timestamp value - again - depending 
   on your selection of `agg_method`.
   NOTE, that this step will very likely change the size of the dataseries 
   to-be-flagged.
   New sampling intervals, covering no data in the original dataseries or only 
   data that got excluded in step (1), will be regarded as representing missing 
   data (Thus get assigned `NaN` value). 
   The original data will be dropped (but can be regained by function 
   `deharmonize`).
4. Depending on the keyword passed to `agg_flag_func`, the original flags get
   aggregated and assigned onto the new, harmonized data, generated in step (3).
   New sampling intervals, covering no data in the original dataseries or only 
   data that got excluded in step (1), will be regarded as representing missing 
   data and thus get assigned the worst flag level available.


`agg_method` keywords:
    
* `"fagg"`: all flags/values in a sampling interval get aggregated with the function passed to `agg_method`
                , and the result gets assigned to the last grid point.
* `"bagg"`: all flags/values in a sampling interval get aggregated with the function passed to `agg_method`
                , and the result gets assigned to the next grid point.
* `"nearest_agg"`: all flags/values in the range (+/- freq/2) of a grid point get 
           aggregated with the function passed to agg_method and assigned to it.


### harmonize_linear2grid

```
harmonize_linear2Grid(freq, flag_assignment_method='nearest_agg', flag_agg_func="max", drop_flags=None)
```
| parameter             | data type         | default value     | description |
| ---------             | ---------         | -------------     | ----------- |
| freq                  | string            |                   | Offset string. Determining the sampling rate of the frequency grid, the data shall be interpolated at.|
| flag_assignment_method| string            | "nearest_agg"     | Method keyword, signifying method used for flags aggregation. See step 4 and table below|
| flag_agg_func         | func              | `"max"`               | String, signifying a function used for flags aggregation. See a table of keywords [here](#aggregations).|   
| drop_flags            | list or Nonetype  | `None`              | Flags to be excluded from harmonization. See description of step 2 below. |

Linear interpolation of an inserted equidistant frequency grid of sampling rate `freq`.

1. All missing values in the data, identified by `np.nan`, 
   get flagged and will be excluded from the aggregation process.
   NOTE, that implicitly this step includes a call to `missing` onto the 
   data-to-be-flagged. 
2. Additionally, if a list is passed to `drop_flags`, all the values in data, 
   that are flagged with a flag, listed in `drop_list`, will be excluded from
   interpolation - meaning, that they will not affect the further 
   aggregation prozess.
3. Data interpolation gets carried out: since the function is a harmonization function, the interpolation will not fill
   gaps in your timeseries, but only calculate an interpolation value for grid points, that are surrounded by 
   valid values within `freq` range. If there is either no valid value to the right, or to the left of a new grid point, 
   that new grid point gets assigned `np.nan` (missing.)
4. Depending on the keyword passed to `flag_assignment_method`, the original flags get
   shifted, or aggregated with `flag_agg_func` onto the new, harmonized data index, generated in step (3).
   New sampling intervals, covering no data in the original dataseries or only 
   data that got excluded in step (1), will be regarded as representing missing 
   data and thus get assigned the worst flag level available.
   

`flag_assignment_method` - Keywords

1. Shifts:
    * `"fshift"`: every grid point gets assigned its ultimately preceeding flag 
      if there is one available in the preceeding sampling interval. If not, BAD - flag gets assigned.
    * `"bshift"`: every grid point gets assigned its first succeeding flag 
      if there is one available in the succeeding sampling interval. If not, BAD - flag gets assigned.
    * `"nearest_shift"`: every grid point gets assigned the flag in its range. ( range = +/- `freq`/2 ).
    * Extra flag fields like "comment", just get shifted along with the flag. 
      Only inserted flags for empty intervals will get signified by the set flag routine of the current flagger.
      Set `set_shift_comment` to `True`,  to apply setFlags signification to all flags.
2. Aggregations:
    * `"fagg"`: all flags in a sampling interval get aggregated with the function passed to `agg_func`
                , and the result gets assigned to the last grid point.
    * `"bagg"`: all flags in a sampling interval get aggregated with the function passed to `agg_func`
                , and the result gets assigned to the next grid point.
    * `"nearest_agg"`: all flags in the range (+/- freq/2) of a grid point get 
                       aggregated with the function passed to `agg_func` and assigned to it.
                      


### harmonize_interpolate2grid

```
harmonize_interpolate2Grid(freq, interpolation_method, interpolation_order=1, flag_assignment_method='nearest_agg', 
                           flag_agg_func="max", drop_flags=None)
```
| parameter             | data type         | default value     | description |
| ---------             | ---------         | -------------     | ----------- |
| freq                  | string            |                   | Offset string. Determining the sampling rate of the frequency grid, the data shall be interpolated at.|
| interpolation_method  | string            |                   | Method keyword, signifying method used for grid interpolation. See step 3 and table below|
| interpolation_order   | integer              | `1`               | If needed - order of the interpolation, carried out.|   
| flag_assignment_method| string            | `"nearest_agg"`   | Method keyword, signifying method used for flags aggregation. See step 4 and table below|
| flag_agg_func         | string              | `"max"`         | String, signifying a function, used for flags aggregation. Must be applicable on the ordered categorical flag type of the current flagger. See a table of keywords [here](#aggregations). |   
| drop_flags            | list or Nonetype  | `None`            | Flags to be excluded from harmonization. See description of step 2 below. |

Interpolation of an inserted equidistant frequency grid of sampling rate `freq`.

1. All missing values in the data, identified by `np.nan`, 
   get flagged and will be excluded from the aggregation process.
   NOTE, that implicitly this step includes a call to `missing` onto the 
   data-to-be-flagged. 
2. Additionally, if a list is passed to `drop_flags`, all the values in data, 
   that are flagged with a flag, listed in `drop_list`, will be excluded from
   interpolation - meaning, that they will not affect the further 
   aggregation prozess.
3. Data interpolation with `interpolation_method` gets carried out: since the function is a harmonization function, the interpolation will not fill
   gaps in your timeseries, but only calculate an interpolation value for grid points, that are surrounded by 
   valid values within `freq` range. If there is either no valid value to the right, or to the left of a new grid point, 
   that new grid point gets assigned `np.nan` (missing.)
4. Depending on the keyword passed to `flag_assignment_method`, the original flags get
   shifted, or aggregated with `flag_agg_func` onto the new, harmonized data index, generated in step (3).
   New sampling intervals, covering no data in the original dataseries or only 
   data that got excluded in step (1), will be regarded as representing missing 
   data and thus get assigned the worst flag level available.

`interpolation_method` - Keywords:
* There are available all the interpolation methods from the pandas.interpolate() method and they can be reffered to with
    the very same keywords, that you would pass to pd.Series.interpolates's method parameter.
* Available interpolations: `"linear"`, `"time"`, `"nearest"`, `"zero"`, `"slinear"`,
    `"quadratic"`, `"cubic"`, `"spline"`, `"barycentric"`, `"polynomial"`, `"krogh"`,
    `"piecewise_polynomial"`, `"spline"`, `"pchip"`, `"akima"`.
* Be careful with pd.Series.interpolate's `"nearest"` and `"pad"`:
      To just fill grid points forward/backward or from the nearest point - and
      assign grid points, that refer to missing data, a nan value, the use of `harmonize_shift2Grid` function is
      recommended, to ensure getting the result expected. (The methods diverge in some 
      special cases and do not properly interpolate grid-only.).

`flag_assignment_method` - Keywords

1. Shifts:
    * `"fshift"`: every grid point gets assigned its ultimately preceeding flag 
      if there is one available in the preceeding sampling interval. If not, BAD - flag gets assigned.
    * `"bshift"`: every grid point gets assigned its first succeeding flag 
      if there is one available in the succeeding sampling interval. If not, BAD - flag gets assigned.
    * `"nearest_shift"`: every grid point gets assigned the flag in its range. ( range = +/- `freq`/2 ).
    * Extra flag fields like "comment", just get shifted along with the flag. 
      Only inserted flags for empty intervals will get signified by the set flag routine of the current flagger.
      Set `set_shift_comment` to `True`,  to apply setFlags signification to all flags.
2. Aggregations:
    * `"fagg"`: all flags in a sampling interval get aggregated with the function passed to `agg_func`
                , and the result gets assigned to the last grid point.
    * `"bagg"`: all flags in a sampling interval get aggregated with the function passed to `agg_func`
                , and the result gets assigned to the next grid point.
    * `"nearest_agg"`: all flags in the range (+/- freq/2) of a grid point get 
                       aggregated with the function passed to `agg_func` and assigned to it.
          


### harmonize_downsample

```
harmonize_downsample(sample_freq, agg_freq, sample_func="mean", agg_func="mean",
                     invalid_flags=None, max_invalid=np.inf)
```
| parameter             | data type         | default value     | description |
| ---------             | ---------         | -------------     | ----------- |
| sample_freq           | string            |                   | Offset String. Determining the intended sampling rate of the data-to-be aggregated |
| agg_freq              | string            |                   | Offset String. Determining the frequency to aggregate to. |
| sample_func           | string or Nonetype  | `"mean"`           | String, signifying a Function to gather/aggregate data within every sampling interval. If `None` is passed, data is expected to already match a sampling grid of `sample_freq`. Additionally to the funcs listed in the agg func table, its possible to pass the keywords `first` and `last`, referring to selection of very first and very last of every sampling intervals meassurement. |   
| agg_func              | string              | `"mean"`           | String, signifying a function used to downsample data from `sample_freq` to `agg_freq`. See a table of keywords [here](#aggregations). |
| invalid_flags         | list or Nonetype  | `None`              | List of flags, to be regarded as signifying invalid values. By default (=`None`), `NaN` data and `BAD`-flagged data is considered invalid. See description below.|   
| max_invalid           | integer           | `Inf`             | Maximum number of invalid data points allowed for an aggregation interval to not get assigned `NaN` |

The function downsamples the data-to-be flagged from its intended sampling rate, assumed to be `sample_freq`, to a lower
sampling rate of `agg_freq`, by applying `agg_func` onto intervals of size `agg_freq`.

If `sample_func` is not `None`, in a preceeding step the data, contained in a sampling interval of `sample_freq`, 
gets aggregated with `sample_func` to a `sampling_freq` sized grid.

The parameter `invalid_flags` allows for marking data values, flagged with a flag listed in `invalid_flags` as invalid.
By setting `max_invalid` to a value < `inf`, you can determine the aggregation of aggregation intervals containing 
more than `max_invalid` invalid values to get assigned `NaN` value.
By default, `BAD` - flagged, as well as missing/ `NaN` data is considered invalid.

Although, the function is a wrapper around `harmonize` - the deharmonization of "real" 
downsamples (`sample_freq` < `agg_freq`) is not recommended, since, the backtracking of flags would result in really 
unexpected results.(BAD - flagging of all  the values contained in an invalid aggregate) 

(an option to just regain initial data frame shape with initial flags is to be implemented)


### harmonize

```
harmonize(freq, inter_method, reshape_method, inter_agg="mean", inter_order=1,
          inter_downcast=False, reshape_agg="max", reshape_missing_flag=None,
          reshape_shift_comment=True, drop_flags=None, 
          data_missing_value=np.nan)
```

| parameter          | data type        | default value    | description |
| ------             | ------           | ------           | ----        |
| freq               | string           |           | Offset string. The frequency of the grid, the data-to-be-flagged shall be projected on.|
| inter_method       | string           |           | A keyword, determining the method, used for projecting the data on the new, equidistant data index. See a list of options below.|
| reshape_method     | string           |           | A keyword, determining the method, used for projecting the flags on the new, equidistant data index. See a list of options below.|
| inter_agg          | string             |`"mean"`  | String, signifying a function, used for aggregation, if an aggregation method is selected as `inter_method`. See a table of keywords [here](#aggregations).|
| inter_order        | int              |`1`        | The order of interpolation applied, if an interpolation method is passed to `inter_method`|
| inter_downcast     | boolean          |`False`    | `True`: Use lower interpolation order to interpolate data chunks that are too short to be interpolated with order `inter_order`. <br/> `False`: Project values of too-short data chunks onto `NaN`. <br/> Option only relevant if `inter_method` can be of certain order.| 
| reshape_agg        | string           | `"max"`   | String, signifying a function, used for aggregation of flags in the interval determined by `reshape_method`. By default (`"max"`), the worst flag gets assigned |
| reshape_missing_flag| string or Nonetype | `None`    | Either a string, referring to a flags name of the flagger you use, or `None`. The flag signified by this parameter gets inserted whenever there is no data available for for an harmonization interval. The default, `None`, leads to insertion of the currents flaggers `BAD` flag.
| reshape_shift_comment | boolean       |`True`     | `True`: Flags that got shifted forward or backward on the new equidistant data index, get resetted additionally. This may, for example, result in eventually present comment fields, to get overwritten with whatever is defaultly been written in this field for the current flagger, if a function sets a flag. <br/> `False`: No reset of the shifted flag will be made. <br/> <br/> Only relevant for flagger having more fields then the flags field and a shifting method passed to `inter_method`|
| drop_flags         | list or Nonetype |`None`     | A list of flags to exclude from harmonization. See step (1) below. If `None` is passed, only BAD - flagged values get dropped. If a list is passed, the BAD flag gets added to that list by default |
| data_missing_value | any valeu        |`np.nan`   | The value, indicating missing data in the dataseries-to-be-flagged.|


The function "harmonizes" the data-to-be-flagged, to match an equidistant 
frequency grid. In general this includes projection and/or interpolation of 
the data at timestamp values, that are multiples of `freq`.

In detail the process includes:

1. All missing values in the data, identified by `data_missing_value` 
   get flagged and will be excluded from the harmonization process.
   NOTE, that implicitly this step includes a call to `missing` onto the 
   data-to-be-flagged. 
2. Additionally, if a list is passed to `drop_flags`, all the values in data, 
   that are flagged with a flag, listed in `drop_list`, will be excluded from
   harmonization - meaning, that they will not affect the further 
   interpolation/aggregation prozess.
3. Depending on the keyword passed to `inter_method`, new data values get 
   calculated for an equidistant timestamp series of frequency `freq`, ranging 
   from start to end of the data-to-be-flagged.
   NOTE, that this step will very likely change the size of the dataseries 
   to-be-flagged.
   New sampling intervals, covering no data in the original dataseries or only 
   data that got excluded in step (1), will be regarded as representing missing 
   data (Thus get assigned `NaN` value and the). 
   The original data will be dropped (but can be regained by function 
   `deharmonize`).
4. Depending on the keyword passed to `reshape_method`, the original flags get
   projected/aggregated onto the new, harmonized data, generated in step (3).
   New sampling intervals, covering no data in the original dataseries or only 
   data that got excluded in step (1), will be regarded as representing missing 
   data and thus get assigned the `reshape_missing` flag.

NOTE, that, if: 

1.  you want to calculate flags on the new, harmonic dataseries and 
    project this flags back onto the original timestamps/flags, you have to 
    add a call to `deharmonize` on this variable in your meta file.

2.  you want to restore the original data shape, as inserted into saqc - you
    have to add a call to deharmonize on all the variables harmonized 
    in the meta. 

Key word overview:

`inter_method` - keywords

1. Shifts:
    * `"fshift"`: every grid point gets assigned its ultimately preceeding value - if there is one available in the preceeding sampling interval.
    * `"bshift"`: every grid point gets assigned its first succeeding value - if there is one available in the succeeding sampling interval.
    * `"nearest_shift"`: every grid point gets assigned the nearest value in its range. ( range = +/- `freq`/2 ).
2. Aggregations:
    * `"fagg"`: all values in a sampling interval get aggregated with the function passed to `agg_method`
                , and the result gets assigned to the last grid point.
    * `"bagg"`: all values in a sampling interval get aggregated with the function passed to `agg_method`
                , and the result gets assigned to the next grid point.
    * `"nearest_agg"`: all values in the range (+/- freq/2) of a grid point get 
                       aggregated with the function passed to agg_method and assigned to it.

3. Interpolations:
    * There are available all the interpolation methods from the pandas.interpolate() method and they can be reffered to with
      the very same keywords, that you would pass to pd.Series.interpolates's method parameter.
    * Available interpolations: `"linear"`, `"time"`, `"nearest"`, `"zero"`, `"slinear"`,
      `"quadratic"`, `"cubic"`, `"spline"`, `"barycentric"`, `"polynomial"`, `"krogh"`,
      `"piecewise_polynomial"`, `"spline"`, `"pchip"`, `"akima"`.
    * If a selected interpolation method needs to get passed an order of 
      interpolation, it will get passed the order, passed to `inter_order`.
    * Note, that ´"linear"´ does not refer to timestamp aware, linear 
      interpolation, but will equally weight every period, no matter how great 
      the covered time gap is. Instead, a timestamp aware, linear interpolation is performed 
      upon ´"time"´ passed as keyword.
    * Be careful with pd.Series.interpolate's `"nearest"` and `"pad"`:
      To just fill grid points forward/backward or from the nearest point - and
      assign grid points, that refer to missing data, a nan value, the use of `"fshift"`, `"bshift"` and `"nearest_shift"` is
      recommended, to ensure getting the result expected. (The methods diverge in some 
      special cases and do not properly interpolate grid-only.).


`reshape_method` - Keywords


1. Shifts:
    * `"fshift"`: every grid point gets assigned its ultimately preceeding flag 
      if there is one available in the preceeding sampling interval. If not, BAD - flag gets assigned.
    * `"bshift"`: every grid point gets assigned its first succeeding flag 
      if there is one available in the succeeding sampling interval. If not, BAD - flag gets assigned.
    * `"nearest_shift"`: every grid point gets assigned the flag in its range. ( range = +/- `freq`/2 ).
    * Extra flag fields like "comment", just get shifted along with the flag. 
      Only inserted flags for empty intervals will get signified by the set flag routine of the current flagger.
      Set `set_shift_comment` to `True`,  to apply setFlags signification to all flags.
2. Aggregations:
    * `"fagg"`: all flags in a sampling interval get aggregated with the function passed to `agg_method`
                , and the result gets assigned to the last grid point.
    * `"bagg"`: all flags in a sampling interval get aggregated with the function passed to `agg_method`
                , and the result gets assigned to the next grid point.
    * `"nearest_agg"`: all flags in the range (+/- freq/2) of a grid point get 
                       aggregated with the function passed to agg_method and assigned to it.


### deharmonize

```
deharmonize(co_flagging)
```
| parameter | data type | default value | description |
| --------- | --------- | ------------- | ----------- |
| co_flagging       | boolean     |               | `False`: depending on the harmonization method applied, only overwrite ultimately preceeding, first succeeding or nearest flag to a harmonized flag. <br/> `True`: Depending on the harmonization method applied, overwrite all the values covered by the succeeding or preceeding sampling intervall, or, all the values in the range of a harmonic flags timestamp. |



After having calculated flags on an equidistant frequency grid, generated by 
a call to a harmonization function, you may want to project
that new flags on to the original data index, or just restore the 
original data shape. Then a call to `deharmonize` will do exactly that.

`deharmonize` will check for harmonization information for the variable it is 
applied on (automatically generated by any call to a harmonization function of that variable) 
and than:

1. Overwrite the harmonized data series with the original dataseries and its timestamps.
2. Project the calculated flags onto the original index, by inverting the 
  flag projection method used for harmonization, meaning, that:
    * if the flags got shifted or aggregated forward, either the flag associated with the ultimatly preceeding 
      original timestamp, to the harmonized flag (`co_flagging`=`False`),
      or all the flags, coverd by the harmonized flags preceeding sampling intervall (`co_flagging`=`True`)
      get overwritten with the harmonized flag - if they are "better" than this harmonized flag. 
      (According to the flagging order of the current flagger.) 
    * if the flags got shifted or aggregated backwards, either the flag associated with the first succeeding 
      original timestamp, to the harmonized flag (`co_flagging`=`False`),
      or all the flags, coverd by the harmonized flags succeeding sampling intervall (`co_flagging`=`True`)
      get overwritten with the harmonized flag - if they are "better" than this harmonized flag. 
      (According to the flagging order of the current flagger.)  
    * if the flags got shifted or aggregated to the nearest harmonic index, 
      either the flag associated with the flag, nearest, to the harmonized flag (`co_flagging`=`False`),
      or all the flags, covered by the harmonized flags range (`co_flagging`=`True`)
      get overwritten with the harmonized flag - if they are "better" than this harmonized flag. 
      (According to the flagging order of the current flagger.)  


### aggregations

Here is a table of aggregation keywords, to pass to the different aggregation parameters, and the functions they refer to. 

| keyword       | function              |
| ---------     | ---------             | 
| `"sum"`       | Sum of values.        |                   
| `"mean"`      | Mean over the values  |                   
| `"min"`       | Minimum               |
| `"max"`       | Maximum               |               
| `"median"`    | Median of the values  |             


## Soil Moisture

### soilMoisture_plateaus

```
soilMoisture_plateaus(plateau_window_min="12h", plateau_var_limit=0.0005,
                      rainfall_window_range="12h", var_total_nans=np.inf, 
                      var_consec_nans=np.inf, derivative_max_lb=0.0025, 
                      derivative_min_ub=0, data_max_tolerance=0.95, 
                      filter_window_size=None, smooth_poly_order=2)
```

| parameter          | data type    | default value | description |
| ------             | ------       | ------        | ----        |
| plateau_window_min | string       | `"12h"`       | Options <br/> - any offset string <br/> <br/> Minimum barrier for the duration, values have to be continouos to be plateau canditaes. See condition (1).|
| plateau_var_limit  | float        | `0.0005`      | Barrier, the variance of a group of values must not exceed to be flagged a plateau. See condition (2). |
| rainfall_range     | string       | `"12h"`       | An Offset string. See condition (3) and (4) |
| var_total_nans     | int or 'inf' | `np.inf`      | Maximum number of nan values allowed, for a calculated variance to be valid. (Default skips the condition.) |
| var_consec_nans    | int or 'inf' | `np.inf`      | Maximum number of consecutive nan values allowed, for a calculated variance to be valid. (Default skips the condition.) |
| derivative_max_lb  | float        | `0.0025`      | Lower bound for the second derivatives maximum in `rainfall_range` range. See condition (3)|
| derivative_min_ub  | float        | `0`           | Upper bound for the second derivatives minimum in `rainfall_range` range. See condition (4)|
| data_max_tolerance | flaot        | `0.95`        | Factor for data max barrier of condition (5).|
| filter_window_size | Nonetype or string   | `None` | Options: <br/> - `None` <br/> - any offset string <br/><br/> Controlls the range of the smoothing window applied with the Savitsky-Golay filter. If None is passed (default), the window size will be two times the sampling rate. (Thus, covering 3 values.) If you are not very well knowing what you are doing - do not change that value. Broader window sizes caused unexpected results during testing phase.|
| smooth_poly_order  | int          | `2` | Order of the polynomial used for fitting while smoothing. |



NOTE, that the dataseries-to-be flagged is supposed to be harmonized to an
equadistant frequency grid.

The function represents a stricter version of the `constant_varianceBased` 
test from the constants detection library. The added constraints for values to 
be flagged (3)-(5), are designed to match the special case of constant value courses of 
soil moisture meassurements and basically check the derivative for being 
determined by preceeding rainfall events ((3) and (4)), as well as the plateau
for being sufficiently high in value (5).

Any set of consecutive values
$`x_k,..., x_{k+n}`$, of a timeseries $`x`$ is flagged, if:

1. $`n > `$`plateau_window_min`
2. $`\sigma(x_k, x_{k+1},..., x_{k+n}) < `$`plateau_var_limit`
3. $`\max(x'_{k-n-s}, x'_{k-n-s+1},..., x'_{k-n+s}) \geq`$ `derivative_max_lb`, with $`s`$ denoting periods per `rainfall_range`  
4. $`\min(x'_{k-n-s}, x'_{k-n-s+1},..., x'_{k-n+s}) \leq`$ `derivative_min_ub`, with $`s`$ denoting periods per `rainfall_range`
5. $`\mu(x_k, x_{k+1},..., x_{k+n}) < \max(x) \times`$`plateau_var_limit`

This Function is an implementation of the soil temperature based Soil Moisture
flagging, as presented in:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture Data
from the international Soil Moisture Network. 2013. Vadoze Zone J.
doi:10.2136/vzj2012.0097.

All parameters default to the values, suggested in this publication.

### SoilMoisture_spikes

```
soilMoisture_spikes(filter_window_size="3h", raise_factor=0.15, dev_cont_factor=0.2,
                   noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar")
```

| parameter          | data type | default value | description |
| ------             | ------    | ------        | ----        |
| filter_window_size | string    | `"3h"`        |             |
| raise_factor       | float     | `0.15`        |             |
| dev_cont_factor    | float     | `0.2`         |             |
| noise_barrier      | integer   | `1`           |             |
| noise_window_size  | string    | `"12h"`       |             |
| noise_statistic    | string    | `"CoVar"`     |             |


The Function is just a wrapper around `flagSpikes_spektrumBased`, from the
spike detection library and performs a call to this function with a parameter
set, referring to:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
Data from the international Soil Moisture Network. 2013.
Vadoze Zone J. doi:10.2136/vzj2012.0097.


### soilMoisture_breaks

```
soilMoisture_breaks(diff_method="raw", filter_window_size="3h",
                   rel_change_rate_min=0.1, abs_change_min=0.01, first_der_factor=10,
                   first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                   scnd_der_ratio_margin_2=10, smooth_poly_order=2)
```

| parameter               | data type | default value | description |
| ------                  | ------    | ------        | ----        |
| diff_method             | string    | `"raw"`       |             |
| filter_window_size      | string    | `"3h"`        |             |
| rel_change_rate_min     | float     | `0.1`         |             |
| abs_change_min          | float     | `0.01`        |             |
| first_der_factor        | integer   | `10`          |             |
| first_der_window_size   | string    | `"12h"`       |             |
| scnd_der_ratio_margin_1 | float     | `0.05`        |             |
| scnd_der_ratio_margin_2 | float     | `10.0`        |             |
| smooth_poly_order       | integer   | `2`           |             |


The Function is just a wrapper around `flagBreaks_spektrumBased`, from the
breaks detection library and performs a call to this function with a parameter
set, referring to:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
Data from the international Soil Moisture Network. 2013.
Vadoze Zone J. doi:10.2136/vzj2012.0097.


### soilMoisture_byFrost

```
soilMoisture_byFrost(soil_temp_reference, tolerated_deviation="1h", frost_level=0)
```

| parameter           | data type | default value | description |
| ------              | ------    | ------        | ----        |
| soil_temp_reference | string    |               |  A string, denoting the fields name in data, that holds the data series of soil temperature values, the to-be-flagged values shall be checked against.|
| tolerated_deviation | string    | `"1h"`        |  An offset string, denoting the maximal temporal deviation, the soil frost states timestamp is allowed to have, relative to the data point to be flagged.|
| frost_level         | integer   | `0`           |  Value level, the flagger shall check against, when evaluating soil frost level. |


The function flags Soil moisture measurements by evaluating the soil-frost-level
in the moment of measurement (+/- `tolerated deviation`).
Soil temperatures below "frost_level" are regarded as denoting frozen soil
state and result in the checked soil moisture value to get flagged.

This Function is an implementation of the soil temperature based Soil Moisture
flagging, as presented in:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture Data
from the international Soil Moisture Network. 2013. Vadoze Zone J.
doi:10.2136/vzj2012.0097.

All parameters default to the values, suggested in this publication.


### soilMoisture_byPrecipitation

```
soilMoisture_byPrecipitation(prec_reference, sensor_meas_depth=0,
                            sensor_accuracy=0, soil_porosity=0,
                            std_factor=2, std_factor_range="24h"
                            ignore_missing=False)
```

| parameter         | data type | default value | description |
| ------            | ------    | ------        | ----        |
| prec_reference    | string    |               | A string, denoting the fields name in data, that holds the data series of precipitation values, the to-be-flagged values shall be checked against.            |
| sensor_meas_depth | integer   | `0`           | Depth of the soil moisture sensor in meter.|
| sensor_accuracy   | integer   | `0`           | Soil moisture sensor accuracy in $`\frac{m^3}{m^{-3}}`$ |
| soil_porosity     | integer   | `0`           | Porosoty of the soil, surrounding the soil moisture sensor |
| std_factor        | integer   | `2`           | See condition (2) |
| std_factor_range  | string    | `"24h"`       | See condition (2) |
| ignore_missing    | bool      | `False`       | If True, the variance of condition (2), will also be calculated if there is a value missing in the time window. Selcting Flase (default) results in values that succeed a time window containing a missing value never being flagged (test not applicable rule) |


Function flags Soil moisture measurements by flagging moisture rises that do not follow up a sufficient
precipitation event. If measurement depth, sensor accuracy of the soil moisture sensor and the porosity of the
surrounding soil is passed to the function, an inferior level of precipitation, that has to preceed a significant
moisture raise within 24 hours, can be estimated. If those values are not delivered, this inferior bound is set
to zero. In that case, any non zero precipitation count will justify any soil moisture raise.

Thus, a data point $`x_k`$ with sampling rate $`f`$ is flagged an invalid soil moisture raise, if:

1. The value to be flagged has to signify a rise. This means, for the quotient $`s = `$ (`raise_reference` / $`f`$):
    * $`x_k > x_{k-s}`$
2. The rise must be sufficient. Meassured in terms of the standart deviation
   $`V`$, of the values in the preceeding `std_factor_range` - window.
   This means, with $`h = `$`std_factor_range` / $`f`$:
    * $`x_k - x_{k-s} >`$ `std_factor` $`\times V(x_{t-h},...,x_k{k})`$
3. Depending on some sensor specifications, there can be calculated a bound $`>0`$, the rainfall has to exceed to justify the eventual soil moisture raise.
   For the series of the precipitation meassurements $`y`$, and the quotient $`j = `$ "24h" /  $`f`$,   this means:
    * $` y_{k-j} + y_{k-j+1} + ... + y_{k} < `$ `sensor_meas_depth` $`\times`$ `sensor_accuracy` $`\times`$ `soil_porosity`


This Function is an implementation of the precipitation based Soil Moisture
flagging, as presented in:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture Data
from the international Soil Moisture Network. 2013. Vadoze Zone J.
doi:10.2136/vzj2012.0097.

All parameters default to the values, suggested in this publication.


## Machine Learning

### machinelearning

```
machinelearning(references, window_values, window_flags, path)
```

| parameter | data type  | default value  | description |
| --------- | ---------- | -------------- | ----------- |
| references    | string or list of strings        |           | the fieldnames of the data series that should be used as reference variables |
| window_values    | integer        |           | Window size that is used to derive the gradients of both the field- and reference-series inside the moving window|
| window_flags   | integer        |          | Window size that is used to count the surrounding automatic flags that have been set before |
| path    | string        |           | Path to the respective model object, i.e. its name and the respective value of the grouping variable. e.g. "models/model_0.2.pkl" |


This Function uses pre-trained machine-learning model objects for flagging. 
This requires training a model by use of the [training script](../ressources/machine_learning/train_machine_learning.py) provided. 
For flagging, inputs to the model are the data of the variable of interest, 
data of reference variables and the automatic flags that were assigned by other 
tests inside SaQC. Therefore, this function should be defined last in the config-file, i.e. it should be the last test that is executed.
Internally, context information for each point is gathered in form of moving 
windows. The size of the moving windows for counting of the surrounding 
automatic flags and for calculation of gradients in the data is specified by 
the user during model training. For the model to work, the parameters 
'references', 'window_values' and 'window_flags' have to be set to the same 
values as during training. For a more detailed description of the modeling 
aproach see the [training script](../ressources/machine_learning/train_machine_learning.py).

