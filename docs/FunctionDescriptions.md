# Implemented QC functions


## `range`

### Signature
```
range(min, max)
```
### Parameters
| parameter | data type | default value | description |
| --------- | --------- | ------------- | ----------- |
| min       | float     |               |             |
| max       | float     |               |             |

### Description


## `missing`

### Signature
```
missing(nodata=NaN)
```

### Parameters
| parameter | data type  | default value  | description |
| --------- | ---------- | -------------- | ----------- |
| nodata    | any        | `NaN`          | Value indicating missing values in the passed data |


### Description
The function flags those values in the the passed data series, that are
associated with "missing" data. The missing data indicator (default: `NaN`), can
be altered to any other value by passing this new value to the parameter `nodata`.


## `seasonalRange`

### Signature
```
sesonalRange(min, max, startmonth=1, endmonth=12, startday=1, endday=31)
```

### Parameters
| parameter  | data type    | default value | description |
| ---------  | -----------  | ----          | ----------- |
| min        | float        |               |             |
| max        | float        |               |             |
| startmonth | integer      | `1`           |             |
| endmonth   | integer      | `12`          |             |
| startday   | integer      | `1`           |             |
| endday     | integer      | `31`          |             |

### Description


## `clear`

### Signature
```
clear()
```

### Parameters
| parameter  | data type    | default value | description |
| ---------  | -----------  | ----          | ----------- |

### Description
Remove all previously set flags.

## `force`


### Signature
```
force()
```


### Parameters
| parameter  | data type    | default value | description |
| ---------  | -----------  | ----          | ----------- |

### Description


## `sliding_outlier`

### Signature
```
sliding_outlier(winsz="1h", dx="1h", count=1, deg=1, z=3.5, method="modZ")
```

### Parameters
| parameter  | data type    | default value | description |
| ---------  | -----------  | ----          | ----------- |
| winsz      | string       | `"1h"`        |             |
| dx         | string       | `"1h"`        |             |
| count      | integer      | `1`           |             |
| deg        | integer      | `1"`          |             |
| z          | float        | `3.5`         |             |
| method     | string       | `"modZ"`      |             |

### Description


## `mad`

### Signature
```
mad(length, z=3.5, freq=None)
```

### Parameters
| parameter  | data type    | default value | description |
| ---------  | -----------  | ----          | ----------- |
| length     |              |               |             |
| z          | float        | `3.5`         |             |
| freq       |              | `None`        |             |


### Description


## `Spikes_Basic`
### Signature
```
Spikes_Basic(thresh, tolerance, window_size)
```

### Parameters
| parameter   | data type | default value | description |
| ------      | ------    | ------        | ----        |
| thresh      | float     |               | Minimum jump margin for spikes. See condition (1). |
| tolerance   | float     |               | Range of area, containing al "valid return values". See condition (2). |
| window_size | ftring    |               | An offset string, denoting the maximal length of "spikish" value courses. See condition (3). |

### Description
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

The implementation is a time-window based version of an outlier test from the
UFZ Python library, that can be found [here](https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py).


## `Spikes_SpektrumBased`

### Signature
```
Spikes_SpektrumBased(raise_factor=0.15, dev_cont_factor=0.2,
                     noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar",
                     smooth_poly_order=2, filter_window_size=None)
```

### Parameters
| parameter          | data type | default value | description |
| ------             | ------    | ------        | ----        |
| raise_factor       | float     | `0.15`        | Minimum change margin for a datapoint to become a candidate for a spike. See condition (1). |
| dev_cont_factor    | float     | `0.2`         | See condition (2). |
| noise_barrier      | float     | `1`           | Upper bound for noisyness of data surrounding potential spikes. See condition (3).|
| noise_window_range | string    | `"12h"`       | Any offset string. Determines the range of the timewindow of the "surrounding" data of a potential spike. See condition (3). |
| noise_statistic    | string    | `"CoVar"`     | Operator to calculate noisyness of data, surrounding potential spike. Either `"Covar"` (=Coefficient od Variation) or `"rvar"` (=relative Variance).|
| smooth_poly_order  | integer   | `2`           | Order of the polynomial fit, applied for smoothing|
| filter_window_size      | Nonetype or string   | `None` | Options: <br/> - `None` <br/> - any offset string <br/><br/> Controlls the range of the smoothing window applied with the Savitsky-Golay filter. If None is passed (default), the window size will be two times the sampling rate. (Thus, covering 3 values.) If you are not very well knowing what you are doing - do not change that value. Broader window sizes caused unexpected results during testing phase.|


### Description
The function detects and flags spikes in input data series by evaluating the
the timeseries' derivatives and applying some conditions to them.

NOTE, that the dataseries-to-be flagged is supposed to be harmonized to an
equadistant frequencie grid.

A datapoint $`x_k `$ of a dataseries $`x`$,
is considered a spike, if:

1. The quotient to its preceeding datapoint exceeds a certain bound:
    * $`|\frac{x_k}{x_{k-1}}| > 1 +`$ `raise_factor`, or:
    * $`|\frac{x_k}{x_{k-1}}| < 1 -`$ `raise_factor`
2. The quotient of the datas second derivate $`x''`$, at the preceeding
   and subsequent timestamps is close enough to 1:
    * $`|\frac{x''_{k-1}}{x''_{k+1}} | > 1 -`$ `dev_cont_factor`, and
    * $`|\frac{x''_{k-1}}{x''_{k+1}} | < 1 +`$ `dev_cont_factor`   
3. The dataset, $`X_k`$, surrounding $`x_{k}`$, within `noise_window_range` range,
   but excluding $`x_{k}`$, is not too noisy. Wheras the noisyness gets measured
   by `noise_statistic`:
    * `noise_statistic`$`(X_k) <`$ `noise_barrier`

NOTE, that the derivative is calculated after applying a savitsky-golay filter
to $`x`$.

This Function is a generalization of the Spectrum based Spike flagging
mechanism as presented in:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
doi:10.2136/vzj2012.0097.


## `constant`

### Signature
```
constant(eps, length, thmin=None)
```

### Parameters
| parameter          | data type | default value | description |
| ------             | ------    | ------        | ----        |
| eps                |           |               |             |
| length             |           |               |             |
| thmin              |           | `None`        |             |

### Description


## `constants_varianceBased`

### Signature
```
constants_varianceBased(plateau_window_min="12h", plateau_var_limit=0.0005,
                        var_total_nans=Inf, var_consec_nans=Inf)
```

### Parameters
| parameter          | data type | default value | description |
| ------             | ------    | ------        | ----        |
| plateau_window_min | string    |               | Options <br/> - any offset string <br/> <br/> Minimum barrier for the duration, values have to be continouos to be plateau canditaes. See condition (1). |
| plateau_var_limit  | float     | `0.0005`      | Barrier, the variance of a group of values must not exceed to be flagged a plateau. See condition (2). |
| var_total_nans     | integer   | `Inf`         | Maximum number of nan values allowed, for a calculated variance to be valid. (Default skips the condition.) |
| var_consec_nans    | integer   | `Inf`         | Maximum number of consecutive nan values allowed, for a calculated variance to be valid. (Default skips the condition.) |


### Description
Function flags plateaus/series of constant values. Any set of consecutive values
$`x_k,..., x_{k+n}`$ of a timeseries $`x`$ is flagged, if:

1. $`n > `$`plateau_window_min`
2. $`\sigma(x_k,..., x_{k+n})`$ < `plateau_var_limit`

NOTE, that the dataseries-to-be flagged is supposed to be harmonized to an
equadistant frequency grid.

NOTE, that when `var_total_nans` or `var_consec_nans` are set to a value < `Inf`
, plateaus that can not be calculated the variance of, due to missing values,
will never be flagged. (Test not applicable rule.)

## `soilMoisture_plateaus`

### Signature
```
soilMoisture_plateaus(plateau_window_min="12h", plateau_var_limit=0.0005,
                      rainfall_window_range="12h", var_total_nans=np.inf, 
                      var_consec_nans=np.inf, derivative_max_lb=0.0025, 
                      derivative_min_ub=0, data_max_tolerance=0.95, 
                      filter_window_size=None, smooth_poly_order=2, **kwargs)
```

### Parameters
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


### Description

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

## `SoilMoistureSpikes`

### Signature
```
SoilMoistureSpikes(filter_window_size="3h", raise_factor=0.15, dev_cont_factor=0.2,
                   noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar")
```

### Parameters
| parameter          | data type | default value | description |
| ------             | ------    | ------        | ----        |
| filter_window_size | string    | `"3h"`        |             |
| raise_factor       | float     | `0.15`        |             |
| dev_cont_factor    | float     | `0.2`         |             |
| noise_barrier      | integer   | `1`           |             |
| noise_window_size  | string    | `"12h"`       |             |
| noise_statistic    | string    | `"CoVar"`     |             |


### Description
The Function is just a wrapper around `flagSpikes_spektrumBased`, from the
spike detection library and performs a call to this function with a parameter
set, referring to:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
Data from the international Soil Moisture Network. 2013.
Vadoze Zone J. doi:10.2136/vzj2012.0097.


## `SoilMoistureBreaks`

### Signature
```
SoilMoistureBreaks(diff_method="raw", filter_window_size="3h",
                   rel_change_rate_min=0.1, abs_change_min=0.01, first_der_factor=10,
                   first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                   scnd_der_ratio_margin_2=10, smooth_poly_order=2)
```

### Parameters
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


### Description
The Function is just a wrapper around `flagBreaks_spektrumBased`, from the
breaks detection library and performs a call to this function with a parameter
set, referring to:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
Data from the international Soil Moisture Network. 2013.
Vadoze Zone J. doi:10.2136/vzj2012.0097.


## `SoilMoistureByFrost`

### Signature
```
SoilMoistureByFrost(soil_temp_reference, tolerated_deviation="1h", frost_level=0)
```

### Parameters
| parameter           | data type | default value | description |
| ------              | ------    | ------        | ----        |
| soil_temp_reference | string    |               |  A string, denoting the fields name in data, that holds the data series of soil temperature values, the to-be-flagged values shall be checked against.|
| tolerated_deviation | string    | `"1h"`        |  An offset string, denoting the maximal temporal deviation, the soil frost states timestamp is allowed to have, relative to the data point to be flagged.|
| frost_level         | integer   | `0`           |  Value level, the flagger shall check against, when evaluating soil frost level. |

### Description

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



## `SoilMoistureByPrecipitation`

### Signature
```
SoilMoistureByPrecipitation(prec_reference, sensor_meas_depth=0,
                            sensor_accuracy=0, soil_porosity=0,
                            std_factor=2, std_factor_range="24h"
                            ignore_missing=False)
```

### Parameters
| parameter         | data type | default value | description |
| ------            | ------    | ------        | ----        |
| prec_reference    | string    |               | A string, denoting the fields name in data, that holds the data series of precipitation values, the to-be-flagged values shall be checked against.            |
| sensor_meas_depth | integer   | `0`           | Depth of the soil moisture sensor in meter.|
| sensor_accuracy   | integer   | `0`           | Soil moisture sensor accuracy in $`\frac{m^3}{m^{-3}}`$ |
| soil_porosity     | integer   | `0`           | Porosoty of the soil, surrounding the soil moisture sensor |
| std_factor        | integer   | `2`           | See condition (2) |
| std_factor_range  | string    | `"24h"`       | See condition (2) |
| ignore_missing    | bool      | `False`       | If True, the variance of condition (2), will also be calculated if there is a value missing in the time window. Selcting Flase (default) results in values that succeed a time window containing a missing value never being flagged (test not applicable rule) |

### Description

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


Function flags Soil moisture measurements by flagging moisture rises that do not follow up a sufficient
precipitation event. If measurement depth, sensor accuracy of the soil moisture sensor and the porosity of the
surrounding soil is passed to the function, an inferior level of precipitation, that has to preceed a significant
moisture raise within 24 hours, can be estimated. If those values are not delivered, this inferior bound is set
to zero. In that case, any non zero precipitation count will justify any soil moisture raise.

This Function is an implementation of the precipitation based Soil Moisture
flagging, as presented in:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture Data
from the international Soil Moisture Network. 2013. Vadoze Zone J.
doi:10.2136/vzj2012.0097.

All parameters default to the values, suggested in this publication.


## `Breaks_SpektrumBased`

### Signature
```                            
Breaks_SpektrumBased(rel_change_min=0.1, abs_change_min=0.01, first_der_factor=10,
                     first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                     scnd_der_ratio_margin_2=10, smooth_poly_order=2,
                     diff_method="raw", filter_window_size="3h")
```

### Parameters
| parameter               | data type | default value | description |
| ------                  | ------    | ------        | ----        |
| rel_change_rate_min     | float     | `0.1`         | Lower bound for the relative difference, a value has to have to its preceeding value, to be a candidate for being break-flagged. See condition (2).|
| abs_change_min          | float     | `0.01`        | Lower bound for the absolute difference, a value has to have to its preceeding value, to be a candidate for being break-flagged. See condition (1).|
| first_der_factor        | float     | `10`          | Factor of the second derivates "arithmetic middle bound". See condition (3).|
| first_der_window_size   | string    | `"12h"`       | Options: <br/> - any offset String <br/> <br/> Determining the size of the window, covering all the values included in the the arithmetic middle calculation of condition (3).|
| scnd_der_ratio_margin_1 | float     | `0.05`        | Range of the area, covering all the values of the second derivatives quotient, that are regarded "sufficiently close to 1" for signifying a break. See condition (5).|
| scnd_der_ratio_margin_2 | float     | `10.0`        | Lower bound for the break succeeding second derivatives quotients. See condition (5). |
| smooth_poly_order       | integer   | `2`           | When calculating derivatives from smoothed timeseries (diff_method="savgol"), this value gives the order of the fitting polynomial calculated in the smoothing process.|
| diff_method             | string    | `"savgol"     | Options: <br/> - `"savgol"`  <br/> - `"raw"` <br/><br/> Select "raw", to skip smoothing before differenciation. |
| filter_window_size      | Nonetype or string   | `None` | Options: <br/> - `None` <br/> - any offset string <br/><br/> Controlls the range of the smoothing window applied with the Savitsky-Golay filter. If None is passed (default), the window size will be two times the sampling rate. (Thus, covering 3 values.) If you are not very well knowing what you are doing - do not change that value. Broader window sizes caused unexpected results during testing phase.|


### Description
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

## `machinelearning`

### Signature
```
machinelearning(references, window_values, window_flags, path)
```

### Parameters
| parameter | data type  | default value  | description |
| --------- | ---------- | -------------- | ----------- |
| references    | string or list of strings        |           | the fieldnames of the data series that should be used as reference variables |
| window_values    | integer        |           | Window size that is used to derive the gradients of both the field- and reference-series inside the moving window|
| window_flags   | integer        |          | Window size that is used to count the surrounding automatic flags that have been set before |
| path    | string        |           | Path to the respective model object, i.e. its name and the respective value of the grouping variable. e.g. "models/model_0.2.pkl" |


### Description
This Function uses pre-trained machine-learning model objects for flagging. 
This requires training a model by use of the [training script](../ressources/machine_learning/train_machine_learning.py) provided. 
For flagging, inputs to the model are the data of the variable of interest, 
data of reference variables and the automatic flags that were assigned by other 
tests inside SaQC. 
Internally, context information for each point is gathered in form of moving 
windows. The size of the moving windows for counting of the surrounding 
automatic flags and for calculation of gradients in the data is specified by 
the user during model training. For the model to work, the parameters 
'references', 'window_values' and 'window_flags' have to be set to the same 
values as during training. For a more detailed description of the modeling 
aproach see the [training script](../ressources/machine_learning/train_machine_learning.py).

## `harmonize`

### Signature
```
harmonize(freq, inter_method, reshape_method, inter_agg=np.mean, inter_order=1,
          inter_downcast=False, reshape_agg=max, reshape_missing_flag=None,
          reshape_shift_comment=True, drop_flags=None, 
          data_missing_value=np.nan)
```

### Parameters
| parameter          | data type        | default value    | description |
| ------             | ------           | ------           | ----        |
| freq               | string           |           | Offset string. The frequency of the grid, the data-to-be-flagged shall be projected on.|
| inter_method       | string           |           | A keyword, determining the method, used for projecting the data on the new, equidistant data index. See a list of options below.|
| reshape_method     | string           |           | A keyword, determining the method, used for projecting the flags on the new, equidistant data index. See a list of options below.|
| inter_agg          | func             |`np.mean`  | A function, used for aggregation, if an aggregation method is selected as `inter_method`. |
| inter_order        | int              |`1`        | The order of interpolation applied, if an interpolation method is passed to `inter_method`|
| inter_downcast     | boolean          |`False`    | `True`: Use lower interpolation order to interpolate data chunks that are too short to be interpolated with order `inter_order`. <br/> `False`: Project values of too-short data chunks onto `NaN`. <br/> Option only relevant if `inter_method` can be of certain order.| 
| reshape_shift_comment | boolean       |`True`     | `True`: Flags that got shifted forward or backward on the new equidistant data index, get resetted additionally. This may, for example, result in eventually present comment fields, to get overwritten with whatever is defaultly been written in this field for the current flagger, if a function sets a flag. <br/> `False`: No reset of the shifted flag will be made. <br/> <br/> Only relevant for flagger having more fields then the flags field and a shifting method passed to `inter_method`|
| drop_flags         | list or Nonetype |`None`     | A list of flags to exclude from harmonization. See step (1) below. If `None` is passed, only BAD - flagged values get dropped. If a list is passed, the BAD flag gets added to that list by default |
| data_missing_value | any valeu        |`np.nan`   | The value, indicating missing data in the dataseries-to-be-flagged.|

### Description

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
   data (Thus get assigned `NaN` value). 
   The original data will be dropped (but can be regained by function 
   `deharmonize`).
4. Depending on the keyword passed to `reshape_method`, the original flags get
   projected/aggregated onto the new, harmonized data, generated in step (3).
   New sampling intervals, covering no data in the original dataseries or only 
   data that got excluded in step (1), will be regarded as representing missing 
   data and thus get assigned the worst flag level available.

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
    * Available interpolations: ´"linear"´, ´"time"´, ´"nearest"´, ´"zero"´, ´"slinear"´,
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
      Only inserted flags for empty intervals will signified by the set flag routine of the current flagger..
      Set `set_shift_comment` to `True`,  to apply setFlags signification to all flags.
2. Aggregations:
    * `"fagg"`: all falgs in a sampling interval get aggregated with the function passed to `agg_method`
                , and the result gets assigned to the last grid point.
    * `"bagg"`: all flags in a sampling interval get aggregated with the function passed to `agg_method`
                , and the result gets assigned to the next grid point.
    * `"nearest_agg"`: all flags in the range (+/- freq/2) of a grid point get 
                       aggregated with the function passed to agg_method and assigned to it.





 
    






