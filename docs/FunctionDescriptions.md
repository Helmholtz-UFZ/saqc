# Implemented QC functions

## range
### Signature
```
range(min, max)
```
### Description


## missing
### Signature
```
missing(nodata=NaN)
```
### Description
The Function flags those values in the the passed data series, that are 
associated with "missing" data. The missing data indicator (`np.nan` by default)
, can be altered to any other value by passing this new value to the 
parameter `nodata`.

## sesonalRange
### Signature
```
sesonalRange(min, max, startmonth=1, endmonth=12, startday=1, endday=31)
```


## clear
### Signature
```
clear()
```
### Description


## force
### Signature
```
force()
```
### Description


## sliding_outlier
### Signature
```
sliding_outlier(winsz="1h", dx="1h", count=1, deg=1, z=3.5, method="modZ")
```
### Description


## mad
### Signature
```
mad(length, z=3.5, freq=None)
```
### Description


## Spikes_Basic
### Signature
```
Spikes_Basic(thresh=7, tol=0, length="15min")
```
### Description
A basic outlier test, that is designed to work for harmonized, as well as raw 
(not-harmonized) data.

The values x(n), x(n+1), .... , x(n+k) of a passed timeseries x, are considered
spikes, if:

1. |x(n-1) - x(n + s)| > `thresh`, for all integers s in {0,1,2,...,k}

2. |x(n-1) - x(n+k+1)| < `tol`

3. |x(n-1).index - x(n+k+1).index| < `length`

By this definition, spikes are values, that, after a jump of margin `thresh`(1), 
are keeping that new value level they jumped to, for a timespan smaller than 
`length` (3), and do then return to the initial value level - 
within a tolerance margin of `tol` (2).  
Note, that this characterization of a "spike", not only includes one-value 
outliers, but also plateau-ish value courses.

The implementation is a time-window based version of an outlier test from the 
UFZ Python library, that can be found here:

https://git.ufz.de/chs/python/blob/master/ufz/level1/spike.py


## Spikes_SpektrumBased
### Signature
```
Spikes_SpektrumBased(filter_window_size="3h", raise_factor=0.15, dev_cont_factor=0.2,
                     noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar",
                     smooth_poly_order=2)
```
### Description

The function detects and flags spikes in input data series by evaluating the 
the timeseries' derivatives and applying some conditions to it. 

NOTE, that the dataseries-to-be flagged is supposed to be harmonized to an 
equadistant frequencie grid.

A datapoint x(k) of a dataseries x, is considered a spike, if:

1. The quotient to its preceeding datapoint exceeds a certain bound:
    * x(k)/x(k-1) > 1 + `raise_factor`, or:
    * x(k)/x(k-1) < 1 - `raise_factor`
2. The quotient of the datas second derivate x'', at the preceeding 
   and subsequent timestamps is close enough to 1:
    * (1 - `dev_cont_factor`) < | x''(k-1)/x''(k+1) |, and
    * (1 + `dev_cont_factor`) > | x''(k-1)/x''(k+1) |   
3. The dataset, surrounding x(k), within `noise_window_size` range, but excluding 
   x(k), is not too noisy. Wheras the noisyness gets measured by 
   `noise_statistic`: 
    * 'noise_statistic'(x.index(k-'noise_window_size'),...,
      x.index(k+'noise_window') < `noise_barrier`


This Function is a generalization of the Spectrum based Spike flagging 
mechanism as presented in:

Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture 
Data from the international Soil Moisture Network. 2013. Vadoze Zone J. 
doi:10.2136/vzj2012.0097.

## constant
### Signature
```
constant(eps, length, thmin=None)
```
### Description


## constants_varianceBased
### Signature
```
constants_varianceBased(plateau_window_min="12h", plateau_var_limit=0.0005,
                        var_total_nans=Inf, var_consec_nans=Inf)
```
### Description


## SoilMoistureSpikes
### Signature
```
SoilMoistureSpikes(filter_window_size="3h", raise_factor=0.15, dev_cont_factor=0.2,
                   noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar")
```
### Description

The Function is just a wrapper around `flagSpikes_SpektrumBased`, from the 
spike detection library and performs a call to this function with a parameter 
set, referring to:

Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture
Data from the international Soil Moisture Network. 2013. 
Vadoze Zone J. doi:10.2136/vzj2012.0097.


## SoilMoistureBreaks
### Signature
```
SoilMoistureBreaks(diff_method="raw", filter_window_size="3h",
                   rel_change_rate_min=0.1, abs_change_min=0.01, first_der_factor=10,
                   first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                   scnd_der_ratio_margin_2=10, smooth_poly_order=2)
```
### Description

The Function is just a wrapper around `flagBreaks_spektrumBased`, from the 
breaks detection library and performs a call to this function with a parameter 
set, referring to:

Dorigo,W,.... Global Automated Quality Control of In Situ Soil Moisture
Data from the international Soil Moisture Network. 2013. 
Vadoze Zone J. doi:10.2136/vzj2012.0097.


## SoilMoistureByFrost
### Signature
```
SoilMoistureByFrost(soil_temp_reference, tolerated_deviation="1h", frost_level=0)
```
### Description



## SoilMoistureByPrecipitation
### Signature
```
SoilMoistureByPrecipitation(prec_reference, sensor_meas_depth=0,
                            sensor_accuracy=0, soil_porosity=0,
                            std_factor=2, std_factor_range="24h")
```
### Description


## Breaks_SpektrumBased
### Signature
```                            
Breaks_SpektrumBased(diff_method="raw", filter_window_size="3h",
                     rel_change_rate_min=0.1, abs_change_min=0.01, first_der_factor=10,
                     first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                     scnd_der_ratio_margin_2=10, smooth_poly_order=2)
```
### Description
