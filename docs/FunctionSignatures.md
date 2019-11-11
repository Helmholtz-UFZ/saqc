# Implemented QC functions
```
range(min, max)
```

```
missing(nodata=NaN)
```

```
sesonalRange(min, max, startmonth=1, endmonth=12, startday=1, endday=31)
```

```
clear()
```

```
force()
```

```
sliding_outlier(winsz="1h", dx="1h", count=1, deg=1, z=3.5, method="modZ")
```

```
mad(length, z=3.5, freq=None)
```

```
Spikes_Basic(thresh=7, tol=0, length="15min")
```

```
Spikes_SpektrumBased(filter_window_size="3h", raise_factor=0.15, dev_cont_factor=0.2,
                     noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar",
                     smooth_poly_order=2)
```

```
constant(eps, length, thmin=None)
```

```
constants_varianceBased(plateau_window_min="12h", plateau_var_limit=0.0005,
                        var_total_nans=Inf, var_consec_nans=Inf)
```

```
SoilMoistureSpikes(filter_window_size="3h", raise_factor=0.15, dev_cont_factor=0.2,
                   noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar")
```

```
SoilMoistureBreaks(diff_method="raw", filter_window_size="3h",
                   rel_change_rate_min=0.1, abs_change_min=0.01, first_der_factor=10,
                   first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                   scnd_der_ratio_margin_2=10, smooth_poly_order=2)
```

```
SoilMoistureByFrost(soil_temp_reference, tolerated_deviation="1h", frost_level=0)
```

```
SoilMoistureByPrecipitation(prec_reference, sensor_meas_depth=0,
                            sensor_accuracy=0, soil_porosity=0,
                            std_factor=2, std_factor_range="24h")
```
                            
```                            
Breaks_SpektrumBased(diff_method="raw", filter_window_size="3h",
                     rel_change_rate_min=0.1, abs_change_min=0.01, first_der_factor=10,
                     first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                     scnd_der_ratio_margin_2=10, smooth_poly_order=2)
```
