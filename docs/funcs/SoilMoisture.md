# Soil Moisture

A collection of soil moisture specific quality check routines.


## Index

- [soilMoisture_spikes](#soilmoisture_spikes)
- [soilMoisture_breaks](#soilmoisture_breaks)
- [soilMoisture_constant](#soilmoisture_constant)
- [soilMoisture_byFrost](#soilmoisture_byfrost)
- [soilMoisture_byPrecipitation](#soilmoisture_byprecipitation)


## soilMoisture_spikes

```
soilMoisture_spikes(raise_factor=0.15, deriv_factor=0.2,
                    noise_func="CoVar", noise_window="12h", noise_thresh=1,
                    smooth_window="3h")
```

| parameter     | default value |
|---------------|---------------|
| raise_factor  | `0.15`        |
| deriv_factor  | `0.2`         |
| noise_thresh  | `1`           |
| noise_window  | `"12h"`       |
| noise_func    | `"CoVar"`     |
| smooth_window | `"3h"`        |


The Function is a wrapper around `flagSpikes_spektrumBased`
with a set of default parameters referring to [1]. For a complete description of 
the algorithm and the available parameters please refer to the documentation of 
[flagSpikes_spektrumBased](docs/funcs/SpikeDetection.md#spikes_spektrumbased)

[1] Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
    Data from the international Soil Moisture Network. 2013.
    Vadoze Zone J. doi:10.2136/vzj2012.0097.


## soilMoisture_breaks

```
soilMoisture_breaks(thresh_rel=0.1, thresh_abs=0.01,
                    first_der_factor=10, first_der_window="12h",
                    scnd_der_ratio_range=0.05, scnd_der_ratio_thresh=10,
                    smooth=False, smooth_window="3h", smooth_poly_deg=2)
```

| parameter             | default value |
|-----------------------|---------------|
| thresh_rel            | `0.1`         |
| thresh_abs            | `0.01`        |
| first_der_factor      | `10`          |
| first_der_window      | `"12h"`       |
| scnd_der_ratio_range  | `0.05`        |
| scnd_der_ratio_thresh | `10.0`        |
| smooth                | `False`       |
| smooth_window         | `"3h"`        |
| smooth_poly_deg       | `2`           |


The Function is a wrapper around `breaks_spektrumBased`
with a set of default parameters referring to [1]. For a complete description of 
the algorithm and the available parameters please refer to the documentation of 
[breaks_spektrumBased](docs/funcs/BreakDetection.md#breaks_spektrumbased).

[1] Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
    Data from the international Soil Moisture Network. 2013.
    Vadoze Zone J. doi:10.2136/vzj2012.0097.


## soilMoisture_constant

```
soilMoisture_constant(window="12h", thresh=0.0005,
                      precipitation_window="12h",
                      tolerance=0.95,
                      deriv_max=0.0025, deriv_min=0,
                      max_missing=None, max_consec_missing=None,
                      smooth_window=None, smooth_poly_deg=2)
```

| parameter            | data type                                                     | default value | description                                                                                                                                                |
|----------------------|---------------------------------------------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| window               | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | Minimum duration during which values need to identical to become plateau candidates. See condition (1)                                                     |
| thresh               | float                                                         | `0.0005`      | Maximum variance of a group of values to still consider them constant. See condition (2)                                                                   |
| precipitation_window | [offset string](docs/ParameterDescriptions.md#offset-strings) | `"12h"`       | See condition (3) and (4)                                                                                                                                  |
| tolerance            | flaot                                                         | `0.95`        | Tolerance factor, see condition (5)                                                                                                                        |
| deriv_min            | float                                                         | `0.0025`      | See condition (3)                                                                                                                                          |
| deriv_max            | float                                                         | `0`           | See condition (4)                                                                                                                                          |
| max_missing          | integer                                                       | `None`        | Maximum number of missing values allowed in `window`, by default this condition is ignored                                                                 |
| max_consec_missing   | integer                                                       | `None`        | Maximum number of consecutive missing values allowed in `window`, by default this condition is ignored                                                     |
| smooth_window        | [offset string](docs/ParameterDescriptions.md#offset-strings) | `None`        | Size of the smoothing window of the Savitsky-Golay filter. The default value `None` results in a window of two times the sampling rate (i.e. three values) |
| smooth_poly_deg      | integer                                                       | `2`           | Degree of the polynomial used for smoothing with the Savitsky-Golay filter                                                                                 |


This function flags plateaus/series of constant values in soil moisture data.

The function represents a stricter version of
[constant_varianceBased](docs/funcs/ConstantDetection.md#constants_variancebased).
The additional constraints (3)-(5), are designed to match the special cases of constant
values in soil moisture meassurements and basically for preceeding precipitation events
(conditions (3) and (4)) and certain plateau level (condition (5)).

Any set of consecutive values
$`x_k,..., x_{k+n}`$, of a timeseries $`x`$ is flagged, if:

1. $`n > `$`window`
2. $`\sigma(x_k, x_{k+1},..., x_{k+n}) < `$`thresh`
3. $`\max(x'_{k-n-s}, x'_{k-n-s+1},..., x'_{k-n+s}) \geq`$ `deriv_min`, with $`s`$ denoting periods per `precipitation_window`
4. $`\min(x'_{k-n-s}, x'_{k-n-s+1},..., x'_{k-n+s}) \leq`$ `deriv_max`, with $`s`$ denoting periods per `precipitation_window`
5. $`\mu(x_k, x_{k+1},..., x_{k+n}) \le \max(x) \cdot`$ `tolerance`

NOTE:
- The time series is expected to be harmonized to an
  [equidistant frequency grid](docs/funcs/TimeSeriesHarmonization.md)

This Function is based on [1] and all default parameter values are taken from this publication.

[1] Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture Data
    from the international Soil Moisture Network. 2013. Vadoze Zone J.
    doi:10.2136/vzj2012.0097.


## soilMoisture_byFrost

```
soilMoisture_byFrost(soil_temp_variable, window="1h", frost_thresh=0)
```

| parameter          | data type                                                     | default value | description                                                   |
|--------------------|---------------------------------------------------------------|---------------|---------------------------------------------------------------|
| soil_temp_variable | string                                                        |               | Name of the soil temperature variable given in the dataset    |
| window             | [offset string](docs/ParameterDescriptions.md#offset-strings) | `"1h"`        | Window around a value checked for frost events                |
| frost_thresh       | float                                                         | `0`           | Soil temperature below `frost_thresh` are considered as frost |


This function flags soil moisture values if the soil temperature
(given in `soil_temp_variable`) drops below `frost_thresh`
within a periode of +/- `window`.

This Function is an implementation of the soil temperature based flagging
as presented in [1] and all default parameter values are taken from this
publication.

[1] Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture Data
    from the international Soil Moisture Network. 2013. Vadoze Zone J.
    doi:10.2136/vzj2012.0097.


## soilMoisture_byPrecipitation

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
