# Implemented QC functions


<!-- this stuff is unused so far 
[->description](docs/FunctionDescriptions.md#harmonize)
[->description](docs/FunctionDescriptions.md#deharmonize)
-->


<!-- 
linkmagic: 
 - the pointing link must be lowercase 
 - for the name everything is fine
 - only use one `#` even if the heading has more `####`
e.g: 
[Facncy Link Name](path/file/#fancy_link_name_all_lowercase)
-->

```
range(min, max)
```
[->description](docs/FunctionDescriptions.md#range)



```
isolated(isolation_range, max_isolated_group_size=1, continuation_range='1min', drop_flags=None)
```
[->description](docs/FunctionDescriptions.md#isolated)



```
missing(nodata=NaN)
```
[->description](docs/FunctionDescriptions.md#missing)



```
sesonalRange(min, max, startmonth=1, endmonth=12, startday=1, endday=31)
```
[->description](docs/FunctionDescriptions.md#seasonalrange)



```
clear()
```
[->description](docs/FunctionDescriptions.md#clear)



```
force()
```
[->description](docs/FunctionDescriptions.md#force)



```
spikes_simpleMad(winsz="1h", length, z=3.5)
```
[->description](docs/FunctionDescriptions.md#spikes_simplemad)



```
spikes_slidingZscore(winsz="1h", dx="1h", count=1, deg=1, z=3.5, method="modZ")
```
[->description](docs/FunctionDescriptions.md#spikes_slidingzscore)



```
spikes_basic(thresh, tolerance, window_size)
```
[->description](docs/FunctionDescriptions.md#spikes_basic)




```
spikes_spektrumBased(raise_factor=0.15, dev_cont_factor=0.2,
                     noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar",
                     smooth_poly_order=2, filter_window_size=None)
```
[->description](docs/FunctionDescriptions.md#spikes_spektrumbased)




```
constant(eps, length, thmin=None)
```
[->description](docs/FunctionDescriptions.md#constant)



```
constants_varianceBased(plateau_window_min="12h", plateau_var_limit=0.0005,
                        var_total_nans=Inf, var_consec_nans=Inf)
```
[->description](docs/FunctionDescriptions.md#constants_varianceBased)




```
soilMoisture_plateaus(plateau_window_min="12h", plateau_var_limit=0.0005,
                      rainfall_window_range="12h", var_total_nans=np.inf, 
                      var_consec_nans=np.inf, derivative_max_lb=0.0025, 
                      derivative_min_ub=0, data_max_tolerance=0.95, 
                      filter_window_size=None, smooth_poly_order=2)
```                      
[->description](docs/FunctionDescriptions.md#soilmoisture_plateaus)




```
soilMoisture_spikes(filter_window_size="3h", raise_factor=0.15, dev_cont_factor=0.2,
                   noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar")
```
[->description](docs/FunctionDescriptions.md#soilmoisturespikes)



```
soilMoisture_breaks(diff_method="raw", filter_window_size="3h",
                   rel_change_rate_min=0.1, abs_change_min=0.01, first_der_factor=10,
                   first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                   scnd_der_ratio_margin_2=10, smooth_poly_order=2)
```
[->description](docs/FunctionDescriptions.md#soilmoisturebreaks)



```
soilMoisture_byFrost(soil_temp_reference, tolerated_deviation="1h", frost_level=0)
```
[->description](docs/FunctionDescriptions.md#soilmoisturebyfrost)




```
soilMoisture_byPrecipitation(prec_reference, sensor_meas_depth=0,
                            sensor_accuracy=0, soil_porosity=0,
                            std_factor=2, std_factor_range="24h")
```
[->description](docs/FunctionDescriptions.md#soilmoisturebyprecipitation)

                            

```                            
breaks_spektrumBased(diff_method="raw", filter_window_size="3h",
                     rel_change_rate_min=0.1, abs_change_min=0.01, first_der_factor=10,
                     first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                     scnd_der_ratio_margin_2=10, smooth_poly_order=2)
```
[->description](docs/FunctionDescriptions.md#breaks_spektrumbased)



```
machinelearning(references, window_values, window_flags, path)
```
[->description](docs/FunctionDescriptions.md#machinelearning)

