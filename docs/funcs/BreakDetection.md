# Break Detection

## Index
[breaks_spektrumBased](#breaks_spektrumbased)

## breaks_spektrumBased

```                            
breaks_spektrumBased(rel_change_min=0.1, abs_change_min=0.01, first_der_factor=10,
                     first_der_window_size="12h", scnd_der_ratio_margin_1=0.05,
                     scnd_der_ratio_margin_2=10, smooth_poly_order=2,
                     diff_method="raw", filter_window_size="3h")
```

| parameter               | data type          | default value | description                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------                  | ------             | ------        | ----                                                                                                                                                                                                                                                                                                                                                                                                               |
| rel_change_rate_min     | float              | `0.1`         | Lower bound for the relative difference, a value has to have to its preceeding value, to be a candidate for being break-flagged. See condition (2).                                                                                                                                                                                                                                                                |
| abs_change_min          | float              | `0.01`        | Lower bound for the absolute difference, a value has to have to its preceeding value, to be a candidate for being break-flagged. See condition (1).                                                                                                                                                                                                                                                                |
| first_der_factor        | float              | `10`          | Factor of the second derivates "arithmetic middle bound". See condition (3).                                                                                                                                                                                                                                                                                                                                       |
| first_der_window_size   | string             | `"12h"`       | Options: <br/> - any offset String <br/> <br/> Determining the size of the window, covering all the values included in the the arithmetic middle calculation of condition (3).                                                                                                                                                                                                                                     |
| scnd_der_ratio_margin_1 | float              | `0.05`        | Range of the area, covering all the values of the second derivatives quotient, that are regarded "sufficiently close to 1" for signifying a break. See condition (5).                                                                                                                                                                                                                                              |
| scnd_der_ratio_margin_2 | float              | `10.0`        | Lower bound for the break succeeding second derivatives quotients. See condition (5).                                                                                                                                                                                                                                                                                                                              |
| smooth_poly_order       | integer            | `2`           | When calculating derivatives from smoothed timeseries (diff_method="savgol"), this value gives the order of the fitting polynomial calculated in the smoothing process.                                                                                                                                                                                                                                            |
| diff_method             | string             | `"savgol"`    | Options: <br/> - `"savgol"`  <br/> - `"raw"` <br/><br/> Select "raw", to skip smoothing before differenciation.                                                                                                                                                                                                                                                                                                    |
| filter_window_size      | Nonetype or string | `None`        | Options: <br/> - `None` <br/> - any offset string <br/><br/> Controlls the range of the smoothing window applied with the Savitsky-Golay filter. If None is passed (default), the window size will be two times the sampling rate. (Thus, covering 3 values.) If you are not very well knowing what you are doing - do not change that value. Broader window sizes caused unexpected results during testing phase. |


The function flags breaks (jumps/drops) by evaluating the derivatives of a time series.

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

NOTE:
- Only works for time series
- The time series is expected to be harmonized to an
  [equidistant frequency grid](docs/funcs/TimeSeriesHarmonization.md)
- The derivatives are calculated after applying a savitsky-golay filter
  to $`x`$.


This Function is a generalization of the spectrum based spike flagging
mechanism as presented in [1].

### References
[1] Dorigo,W. et al.: Global Automated Quality Control of In Situ Soil Moisture
    Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
    doi:10.2136/vzj2012.0097.

