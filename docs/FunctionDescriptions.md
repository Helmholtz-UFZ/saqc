# Implemented Quality Check Functions

Index of the main documentation of the implemented functions, their purpose and parametrization.

## Index

- [Miscellaneous](docs/funcs/Miscellaneous.md)
  - [range](docs/funcs/Miscellaneous.md#range)
  - [seasonalRange](docs/funcs/Miscellaneous.md#seasonalrange)
  - [isolated](docs/funcs/Miscellaneous.md#isolated)
  - [missing](docs/funcs/Miscellaneous.md#missing)
  - [clear](docs/funcs/Miscellaneous.md#clear)
  - [force](docs/funcs/Miscellaneous.md#force)
- [Spike Detection](docs/funcs/SpikeDetection.md)
  - [spikes_basic](docs/funcs/SpikeDetection.md#spikes_basic)
  - [spikes_simpleMad](docs/funcs/SpikeDetection.md#spikes_simplemad)
  - [spikes_slidingZscore](docs/funcs/SpikeDetection.md#spikes_slidingzscore)
  - [spikes_spektrumBased](docs/funcs/SpikeDetection.md#spikes_spektrumbased)
- [Constant Detection](docs/funcs/ConstantDetection.md)
  - [constant](docs/funcs/ConstantDetection.md#constant)
  - [constants_varianceBased](docs/funcs/ConstantDetection.md#constants_variancebased)
- [Break Detection](docs/funcs/BreakDetection.md)
  - [breaks_spektrumBased](docs/funcs/BreakDetection.md#breaks_spektrumbased)
- [Time Series Harmonization](docs/funcs/TimeSeriesHarmonization.md)
  - [harmonize_shift2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_shift2grid)
  - [harmonize_aggregate2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_aggregate2grid)
  - [harmonize_linear2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_linear2grid)
  - [harmonize_interpolate2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_interpolate2grid)
  - [harmonize_downsample](docs/funcs/TimeSeriesHarmonization.md#harmonize_downsample)
  - [harmonize](docs/funcs/TimeSeriesHarmonization.md#harmonize)
  - [deharmonize](docs/funcs/TimeSeriesHarmonization.md#deharmonize)
- [Soil Moisture](#soil-moisture)
  - [soilMoisture_plateaus](#soilmoisture_plateaus)
  - [soilMoisture_spikes](#soilmoisture_spikes)
  - [soilMoisture_breaks](#soilmoisture_breaks)
  - [soilMoisture_byFrost](#soilmoisture_byfrost)
  - [soilMoisture_byPrecipitation](#soilmoisture_byprecipitation)
- [Machine Learning](#machine-learning)
  - [machinelearning](#machinelearning)



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

