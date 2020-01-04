# Time Series Harmonization

A collection of functions to harmonize time series.

## Index

[harmonize_shift2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_shift2grid)
[harmonize_aggregate2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_aggregate2grid)
[harmonize_linear2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_linear2grid)
[harmonize_interpolate2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_interpolate2grid)
[harmonize_downsample](docs/funcs/TimeSeriesHarmonization.md#harmonize_downsample)
[harmonize](docs/funcs/TimeSeriesHarmonization.md#harmonize)
[deharmonize](docs/funcs/TimeSeriesHarmonization.md#deharmonize)


## harmonize_shift2grid

```
harmonize_shift2Grid(freq, shift_method='nearest_shift', drop_flags=None)
```
| parameter    | data type        | default value   | description                                                                 |
|--------------|------------------|-----------------|-----------------------------------------------------------------------------|
| freq         | string           |                 | Offset string. Detemining the frequency grid, the data shall be shifted to. |
| shift_method | string           | `nearest_shift` | Method, used for shifting of data and flags. See a list of methods below.   |
| drop_flags   | list or Nonetype | `None`          | Flags to be excluded from harmonization. See description of step 3 below.   |


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
| parameter     | data type        | default value | description                                                                                              |
|---------------|------------------|---------------|----------------------------------------------------------------------------------------------------------|
| freq          | string           |               | Offset string. Determining the sampling rate of the frequency grid, the data shall be aggregated to.     |
| agg_func      | string           |               | String, signifying a function used for data aggregation. See a table of keywords [here](#aggregations).  |
| agg_method    | string           | `nearest_agg` | Method, determining the range of data and flags aggregation. See a list of methods below.                |
| flag_agg_func | string           | `"max"`       | String, signifying a function used for flags aggregation. See a table of keywords [here](#aggregations). |
| drop_flags    | list or Nonetype | `None`        | Flags to be excluded from harmonization. See description of step 2 below.                                |


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


## harmonize_linear2grid

```
harmonize_linear2Grid(freq, flag_assignment_method='nearest_agg', flag_agg_func="max", drop_flags=None)
```
| parameter              | data type        | default value | description                                                                                              |
|------------------------|------------------|---------------|----------------------------------------------------------------------------------------------------------|
| freq                   | string           |               | Offset string. Determining the sampling rate of the frequency grid, the data shall be interpolated at.   |
| flag_assignment_method | string           | "nearest_agg" | Method keyword, signifying method used for flags aggregation. See step 4 and table below                 |
| flag_agg_func          | func             | `"max"`       | String, signifying a function used for flags aggregation. See a table of keywords [here](#aggregations). |
| drop_flags             | list or Nonetype | `None`        | Flags to be excluded from harmonization. See description of step 2 below.                                |

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



## harmonize_interpolate2grid

```
harmonize_interpolate2Grid(freq, interpolation_method, interpolation_order=1, flag_assignment_method='nearest_agg',
                           flag_agg_func="max", drop_flags=None)
```
| parameter              | data type        | default value   | description                                                                                                                                                                               |
|------------------------|------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| freq                   | string           |                 | Offset string. Determining the sampling rate of the frequency grid, the data shall be interpolated at.                                                                                    |
| interpolation_method   | string           |                 | Method keyword, signifying method used for grid interpolation. See step 3 and table below                                                                                                 |
| interpolation_order    | integer          | `1`             | If needed - order of the interpolation, carried out.                                                                                                                                      |
| flag_assignment_method | string           | `"nearest_agg"` | Method keyword, signifying method used for flags aggregation. See step 4 and table below                                                                                                  |
| flag_agg_func          | string           | `"max"`         | String, signifying a function, used for flags aggregation. Must be applicable on the ordered categorical flag type of the current flagger. See a table of keywords [here](#aggregations). |
| drop_flags             | list or Nonetype | `None`          | Flags to be excluded from harmonization. See description of step 2 below.                                                                                                                 |

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


## harmonize_downsample

```
harmonize_downsample(sample_freq, agg_freq, sample_func="mean", agg_func="mean",
                     invalid_flags=None, max_invalid=np.inf)
```
| parameter     | data type          | default value | description                                                                                                                                                                                                                                                                                                                                                                                |
|---------------|--------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| sample_freq   | string             |               | Offset String. Determining the intended sampling rate of the data-to-be aggregated                                                                                                                                                                                                                                                                                                         |
| agg_freq      | string             |               | Offset String. Determining the frequency to aggregate to.                                                                                                                                                                                                                                                                                                                                  |
| sample_func   | string or Nonetype | `"mean"`      | String, signifying a Function to gather/aggregate data within every sampling interval. If `None` is passed, data is expected to already match a sampling grid of `sample_freq`. Additionally to the funcs listed in the agg func table, its possible to pass the keywords `first` and `last`, referring to selection of very first and very last of every sampling intervals meassurement. |
| agg_func      | string             | `"mean"`      | String, signifying a function used to downsample data from `sample_freq` to `agg_freq`. See a table of keywords [here](#aggregations).                                                                                                                                                                                                                                                     |
| invalid_flags | list or Nonetype   | `None`        | List of flags, to be regarded as signifying invalid values. By default (=`None`), `NaN` data and `BAD`-flagged data is considered invalid. See description below.                                                                                                                                                                                                                          |
| max_invalid   | integer            | `Inf`         | Maximum number of invalid data points allowed for an aggregation interval to not get assigned `NaN`                                                                                                                                                                                                                                                                                        |

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


## harmonize

```
harmonize(freq, inter_method, reshape_method, inter_agg="mean", inter_order=1,
          inter_downcast=False, reshape_agg="max", reshape_missing_flag=None,
          reshape_shift_comment=True, drop_flags=None,
          data_missing_value=np.nan)
```

| parameter             | data type          | default value | description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-----------------------|--------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| freq                  | string             |               | Offset string. The frequency of the grid, the data-to-be-flagged shall be projected on.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| inter_method          | string             |               | A keyword, determining the method, used for projecting the data on the new, equidistant data index. See a list of options below.                                                                                                                                                                                                                                                                                                                                                                         |
| reshape_method        | string             |               | A keyword, determining the method, used for projecting the flags on the new, equidistant data index. See a list of options below.                                                                                                                                                                                                                                                                                                                                                                        |
| inter_agg             | string             | `"mean"`      | String, signifying a function, used for aggregation, if an aggregation method is selected as `inter_method`. See a table of keywords [here](#aggregations).                                                                                                                                                                                                                                                                                                                                              |
| inter_order           | int                | `1`           | The order of interpolation applied, if an interpolation method is passed to `inter_method`                                                                                                                                                                                                                                                                                                                                                                                                               |
| inter_downcast        | boolean            | `False`       | `True`: Use lower interpolation order to interpolate data chunks that are too short to be interpolated with order `inter_order`. <br/> `False`: Project values of too-short data chunks onto `NaN`. <br/> Option only relevant if `inter_method` can be of certain order.                                                                                                                                                                                                                                |
| reshape_agg           | string             | `"max"`       | String, signifying a function, used for aggregation of flags in the interval determined by `reshape_method`. By default (`"max"`), the worst flag gets assigned                                                                                                                                                                                                                                                                                                                                          |
| reshape_missing_flag  | string or Nonetype | `None`        | Either a string, referring to a flags name of the flagger you use, or `None`. The flag signified by this parameter gets inserted whenever there is no data available for for an harmonization interval. The default, `None`, leads to insertion of the currents flaggers `BAD` flag.                                                                                                                                                                                                                     |
| reshape_shift_comment | boolean            | `True`        | `True`: Flags that got shifted forward or backward on the new equidistant data index, get resetted additionally. This may, for example, result in eventually present comment fields, to get overwritten with whatever is defaultly been written in this field for the current flagger, if a function sets a flag. <br/> `False`: No reset of the shifted flag will be made. <br/> <br/> Only relevant for flagger having more fields then the flags field and a shifting method passed to `inter_method` |
| drop_flags            | list or Nonetype   | `None`        | A list of flags to exclude from harmonization. See step (1) below. If `None` is passed, only BAD - flagged values get dropped. If a list is passed, the BAD flag gets added to that list by default                                                                                                                                                                                                                                                                                                      |
| data_missing_value    | any valeu          | `np.nan`      | The value, indicating missing data in the dataseries-to-be-flagged.                                                                                                                                                                                                                                                                                                                                                                                                                                      |


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


## deharmonize

```
deharmonize(co_flagging)
```

| parameter   | data type | default value | description                                                                                                                                                                                                                                                                                                                                                           |
|-------------|-----------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| co_flagging | boolean   |               | `False`: depending on the harmonization method applied, only overwrite ultimately preceeding, first succeeding or nearest flag to a harmonized flag. <br/> `True`: Depending on the harmonization method applied, overwrite all the values covered by the succeeding or preceeding sampling intervall, or, all the values in the range of a harmonic flags timestamp. |


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


## Aggregation Functions

Here is a table of aggregation keywords, to pass to the different aggregation parameters, and the functions they refer to.

| keyword    | function             |
|------------|----------------------|
| `"sum"`    | Sum of values.       |
| `"mean"`   | Mean over the values |
| `"min"`    | Minimum              |
| `"max"`    | Maximum              |
| `"median"` | Median of the values |
