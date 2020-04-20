# Time Series Harmonization

A collection of functions to harmonize time series.

## Index

- [harm_shift2Grid](#harm_shift2grid)
- [harm_aggregate2Grid](#harm_aggregate2grid)
- [harm_linear2Grid](#harm_linear2grid)
- [harm_interpolate2Grid](#harm_interpolate2grid)
- [harm_downsample](#harm_downsample)
- [harm_harmonize](#harm_harmonize)
- [harm_deharmonize](#harm_deharmonize)


## harm_shift2grid

```
harm_shift2Grid(freq, method='nshift')
```
| parameter | data type                                                     | default value | description                           |
|-----------|---------------------------------------------------------------|---------------|---------------------------------------|
| freq      | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | Frequency of the target grid          |
| method    | [method string](#shift-methods)                               | `"nshift"`    | Method used to shift values and flags |


The function "harmonizes" a time series to an equidistant frequency
grid by shifting data points to multiples of `freq`.

This process includes:

1. All missing values in the data set get [flagged](docs/funcs/Miscellaneous-md#flagmissing). 
   These values will be excluded from the shifting process.
2. Depending on the `method`, the data points and the associated
   flags will be assigned to a timestamp in the target grid
   
NOTE:
- The data will be projected to an regular grid ranging from 
  the first to the last timestamp of the original time series
- Because of the above, the size of the harmonized time series
  is likely to differ from the size of the original series
- Data from the original time series might be dropped 
  (e.g. if there are multiple candidates for a shift, only 
  one is used), but can be restored by [harm_deharmonize](#harm_deharmonize)

## harm_aggregate2grid

```
harm_aggregate2Grid(freq, value_func, flag_func="max", method='nagg')
```
| parameter  | data type                                                     | default value | description                                     |
|------------|---------------------------------------------------------------|---------------|-------------------------------------------------|
| freq       | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | Frequency of the target grid                    |
| value_func | [function string](#aggregation-functions)                     |               | Function used for data aggregation              |
| flag_func  | [function string](#aggregation-functions)                     | `"max"`       | Function used for flags aggregation             |
| method     | [method string](#aggregation-methods)                         | `"nagg"`      | Method used to assign values to the target grid |


The function "harmonizes" a time series to an equidistant frequency grid
by aggregating data points to multiples of `freq` using the `method`.

This process includes:

1. All missing values in the data set get [flagged](docs/funcs/Miscellaneous-md#flagmissing). 
   These values will be excluded from the aggregation process
2. Values and flags will be aggregated by `value_func` and `flag_func` respectively
3. Depending on the `method`, the aggregation results will be assigned to a timestamp
   in the target grid

NOTE:
- The data will be projected to an regular grid ranging from 
  the first to the last timestamp of the original time series
- Because of the above, the size of the harmonized time series
  is likely to differ from the size of the original series
- Newly introduced intervals not covering any data in the original
  dataset will be treated as missing data


## harm_linear2grid

```
harm_linear2Grid(freq, method='nagg', func="max")
```

| parameter | data type                                                                 | default value | description                                       |
|-----------|---------------------------------------------------------------------------|---------------|---------------------------------------------------|
| freq      | [offset string](docs/ParameterDescriptions.md#offset-strings)             |               | Frequency of the target grid                      |
| method    | [shift](#shift-methods)/[aggregation](#aggregation-methods) method string | `"nagg"`      | Method used to propagate flags to the target grid |
| func      | [function string](#aggregation-functions)                                 | `"max"`       | Function used for flags aggregation               |


The function "harmonizes" a time series to an equidistant frequency grid
by linear interpolation of data points to multiples of `freq`.

This process includes:

1. All missing values in the data set get [flagged](docs/funcs/Miscellaneous-md#flagmissing). 
   These values will be excluded from the aggregation process
2. Linear interpolation. This is not a gap filling algorithm, only target grid points, 
   that are surrounded by valid data points in the original data set within a range 
   of `freq` will be calculated.
4. Depending on the `method`, the original flags get shifted
   or aggregated with `func` to the target grid


NOTE:
- Newly introduced intervals not covering any data in the original
  dataset will be treated as missing data


## harm_interpolate2grid

```
harm_interpolate2Grid(freq,
                      method, order=1,
                      flag_method='nagg', flag_func="max")
```
| parameter   | data type                                                                 | default value | description                                                             |
|-------------|---------------------------------------------------------------------------|---------------|-------------------------------------------------------------------------|
| freq        | [offset string](docs/ParameterDescriptions.md#offset-strings)             |               | Frequency of the target grid                                            |
| method      | [interpolation method string](#interpolation-methods)                     |               | Interpolation method                                                    |
| order       | integer                                                                   | `1`           | Order of the interpolation, only relevant if applicable in the `method` |
| flag_method | [shift](#shift-methods)/[aggregation](#aggregation-methods) method string | `"nagg"`      | Method used to propagate flags to the target grid                       |
| flag_func   | [function string](#aggregation-functions)                                 | `"max"`       | Function used for flags aggregation                                     |


The function "harmonizes" a time series to an equidistant frequency grid
by interpolation of data points to multiples of `freq`.

This process includes:

1. All missing values in the data set get [flagged](docs/funcs/Miscellaneous-md#flagmissing). 
   These values will be excluded from the aggregation process
2. Interpolation with `method`. This is not a gap filling algorithm,
   only target grid points, that are surrounded by valid data points in the original
   data set within a range of `freq` will be calculated.
3. Depending on the `method`, the original flags get shifted
   or aggregated with `func` to the target grid

NOTE:
- Newly introduced intervals not covering any data in the original
  dataset will be treated as missing data
- We recommended `harmonize_shift2Grid` over the `method`s
  `nearest` and `pad`


## harm_downsample

```
harm_downsample(sample_freq, agg_freq,
                sample_func="mean", agg_func="mean",
                max_invalid=None)
```
| parameter   | data type                                                     | default value | description                                                                                                                    |
|-------------|---------------------------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------|
| sample_freq | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | Frequency of the source grid                                                                                                   |
| agg_freq    | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | Frequency of the target grid                                                                                                   |
| sample_func | [function string](#aggregation-function)                      | `"mean"`      | Function used to aggregate data to `sample_freq`. If `None` the data is expected to have a frequency of `sample_freq`          |
| agg_func    | [function string](#aggregation-function)                      | `"mean"`      | Function used to aggregate data from `sample_freq` to `agg_freq`                                                               |
| max_invalid | integer                                                       | `None`        | If the number of invalid data points (missing/flagged) within an aggregation interval exceeds `max_invalid` it is set to `NAN` |

The function downsamples a time series from its `sample_freq` to the lower
sampling rate `agg_freq`, by aggregation with `agg_func`.

If a `sample_func` is given, the data will be aggregated to `sample_freq`
before downsampling.

NOTE:
- Although the function is a wrapper around `harm_harmonize`, the deharmonization of "true"
  downsamples (`sample_freq` < `agg_freq`) is not supported yet.


## harm_harmonize

```
harm_harmonize(freq, inter_method, reshape_method, inter_agg="mean", inter_order=1,
               inter_downcast=False, reshape_agg="max", reshape_missing_flag=None,
               reshape_shift_comment=False, data_missing_value=np.nan)
```

| parameter             | data type                                                                                                         | default value | description                                                                                                                                                                                                                                   |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| freq                  | [offset string](docs/ParameterDescriptions.md#offset-strings)                                                     |               | Frequency of the target grid                                                                                                                                                                                                                  |
| inter_method          | [shift](#shift-methods)/[aggregation](#aggregation-methods)/[interpolation](#interpolation-methods) method string |               | Method used to project values to the target grid                                                                                                                                                                                              |
| reshape_method        | [shift](#shift-methods)/[aggregation](#aggregation-methods) method string                                         |               | Method used to project flags to the target grid                                                                                                                                                                                               |
| inter_agg             | [aggregation function string](#aggregation-functions)                                                             | `"mean"`      | Function used for aggregation, if an `inter_method` is given                                                                                                                                                                                  |
| inter_order           | int                                                                                                               | `1`           | The order of interpolation applied, if an `inter_method` is given                                                                                                                                                                             |
| inter_downcast        | bool                                                                                                              | `False`       | `True`: Decrease interpolation order if data chunks that are too short to be interpolated with order `inter_order`. <br/> `False`: Project those data chunks to `NAN`. <br/> Option only relevant if `inter_method` supports an `inter_order` |
| reshape_agg           | [aggregation function string](#aggregation-functions)                                                             | `"max"`       | Function used for the aggregation of flags. By default (`"max"`) the worst/highest flag is assigned                                                                                                                                           |
| reshape_missing_flag  | string                                                                                                            | `None`        | Valid flag value, that will be used for empty harmonization intervals. By default (`None`) such intervals are set to `BAD`                                                                                                                    |
| reshape_shift_comment | bool                                                                                                              | `False`       | `True`: Shifted flags will be reset, other fields associated with a flag might get lost. <br/> `False`: Shifted flags will not be reset. <br/> <br/> Only relevant for multi-column flagger and a given `inter_method`                        |
| data_missing_value    | Any                                                                                                               | `np.nan`      | The value, indicating missing data                                                                                                                                                                                                            |


The function "harmonizes" a time series to an equidistant frequency grid.
In general this includes projection and/or interpolation of the data to
timestamps, that are multiples of `freq`.

This process includes:

1. All missing values equal to `data_missing_value` in the data set
   get [flagged](docs/funcs/Miscellaneous-md#flagmissing). 
   These values will be excluded from the aggregation process
2. Values will be calculated according to the given `inter_method`
3. Flags will be calculated according to the given `reshape_method`

NOTE:
- The data will be projected to an regular grid ranging from 
  the first to the last timestamp of the original time series
- Because of the above, the size of the harmonized time series
  is likely to differ from the size of the original series
- Newly introduced intervals not covering any data in the original
  dataset will be set to `data_missing_value` and `reshape_missing`
  respectively
- Data from the original time series might be dropped, but can
  be restored by [deharmonize](#deharmonize)
- Flags calculated on the new harmonized data set can be projected
  to the original grid by [harm_deharmonize](#harm_deharmonize)


## harm_deharmonize

```
harm_deharmonize(co_flagging=False)
```

| parameter   | data type | default value | description                                                    |
|-------------|-----------|---------------|----------------------------------------------------------------|
| co_flagging | boolean   | `False`       | Control the bahviour of the flag reprojection, see description |


This functions projects harmonized datasets back to their original time stamps
and thereby restores the original data shape.

A combination of calls to one of the `harm_*` functions and `harm_deharmonize`,
allows to leverage information from data sets with differing timestamps/frequencies
and bring the generated information back to the original dataset.

`_harm_deharmonize` will implicitly revert the methods and functions applied during
harmonization. I.e.:
- The harmonized time series will be dropped in favor of the original one
- Flags are projected to the original time stamps if the are 'worse'/higher
  than the original. The direction of this projection is invert to the
  shift/aggregation direction in `harm_*`, i.e. a forward shift in
  `harm_*` will result in a backward shift in `harm_deharmonize` and vice
   versa.
- The projection behavior is controlled by the value of `co_flagging`:
  + `False`: Project a flag from the harmonized time series to a single 
     flag in the deharmonized data set
  + `True`: Project a flag in the harmonized time series to all flags 
     in the respective projection interval.
     
  Let's say during harmonization a dataset was aggregated to a lower 
  frequency (e.g. a time series with a frequency of 10 minutes was 
  resampled to one with a frequency of 1 hour) and needs to be deharmonized.
  If `co_flagging` is `True`, the flags from the harmonized dataset 
  will be projected to all the six values within the aggregation period,
  if `co_flagging` is False, only the next/last/nearest value in the 
  deharmonized dataset will inherit the flag from the harmonized 
  time series.
  

## Parameter Descriptions

### Aggregation Functions

| keyword    | description                   |
|------------|-------------------------------|
| `"sum"`    | sum of the values             |
| `"mean"`   | arithmetic mean of the values |
| `"min"`    | minimum value                 |
| `"max"`    | maximum value                 |
| `"median"` | median of the values          |
| `"first"`  | first value                   |
| `"last"`   | last value                    |

### Aggregation Methods

| keyword  | description                                                       |
|----------|-------------------------------------------------------------------|
| `"fagg"` | aggregation result is propagated to the next target grid point    |
| `"bagg"` | aggregation result is propagated to the last target grid point    |
| `"nagg"` | aggregation result is propagated to the closest target grid point |


### Shift Methods

| keyword    | description                                                                    |
|------------|--------------------------------------------------------------------------------|
| `"fshift"` | propagate the last valid value/flag to the grid point or fill with `BAD`/`NAN` |
| `"bshift"` | propagate the next valid value/flag to the grid point or fill with `BAD`/`NAN` |
| `"nshift"` | propagate the closest value/flag to the grid point or fill with `BAD`/`NAN`    |


### Interpolation Methods

- All the `pandas.Series` [interpolation methods](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html)
  are supported
- Available interpolations:
  + `"linear"`
  + `"time"`
  + `"nearest"`
  + `"zero"`
  + `"slinear"`
  + `"quadratic"`
  + `"cubic"`
  + `"spline"`
  + `"barycentric"`
  + `"polynomial"`
  + `"krogh"`
  + `"piecewise_polynomial"`
  + `"spline"`
  + `"pchip"`
  + `"akima"`
