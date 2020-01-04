# Break Detection

## Index
[breaks_spektrumBased](#breaks_spektrumbased)

## breaks_spektrumBased

```                            
breaks_spektrumBased(thresh_rel=0.1, thresh_abs=0.01,
                     first_der_factor=10, first_der_window="12h",
                     scnd_der_ratio_range=0.05, scnd_der_ratio_thresh=10,
                     smooth=True, smooth_window="3h", smooth_poly_deg=2,)
```

| parameter             | data type                                                     | default value | description                                                                                                                                                           |
|-----------------------|---------------------------------------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| thresh_rel            | float                                                         | `0.1`         | Minimum relative difference between two values to consider the latter as a break candidate. See condition (1)                                                         |
| thresh_abs            | float                                                         | `0.01`        | Minimum absolute difference between two values to consider the latter as a break candidate. See condition (2)                                                         |
| first_der_factor      | float                                                         | `10`          | Multiplication factor for arithmetic mean of the first derivatives surrounding a break candidate. See condition (3).                          |
| first_der_window      | [offset string](docs/ParameterDescriptions.md#offset-strings) | `"12h"`       | Window around a break candidate for which the arithmetic mean is calculated. See condition (3)                                                          |
| scnd_der_ratio_range  | float                                                         | `0.05`        | Range of the area, covering all the values of the second derivatives quotient, that are regarded "sufficiently close to 1" for signifying a break. See condition (5). |
| scnd_der_ratio_thresh | float                                                         | `10.0`        | Threshold for the ratio of the second derivatives succeeding a break. See condition (5).                                                                              |
| smooth                | bool                                                          | `True`        | Smooth the time series before differentiation using the Savitsky-Golay filter                                                                                          |
| smooth_window         | [offset string](docs/ParameterDescriptions.md#offset-strings) | `None`        | Size of the smoothing window of the Savitsky-Golay filter. The default value `None` results in a window of two times the sampling rate (i.e. three values)            |
| smooth_poly_deg       | integer                                                       | `2`           | Degree of the polynomial used for smoothing with the Savitsky-Golay filter                                                                                            |


The function flags breaks (jumps/drops) by evaluating the derivatives of a time series.

A value $`x_k`$ of a time series $`x_t`$ with timestamps $`t_i`$, is considered to be a break, if:

1. $`x_k`$ represents a sufficiently large relative jump:
    * $`|\frac{x_k - x_{k-1}}{x_k}| >`$ `thresh_rel`
2. $`x_k`$ represents a sufficient absolute jump:
    * $`|x_k - x_{k-1}| >`$ `thresh_abs`
3. The dataset $`X = x_i, ..., x_{k-1}, x_{k+1}, ..., x_j`$, with  
   $`|t_{k-1} - t_i| = |t_j - t_{k+1}| =`$ `first_der_window` fulfills the following condition:
   
   $`|x'_k| >`$ `first_der_factor` $` \times \bar{X} `$
   
   where $`\bar{X}`$ denotes the arithmetic mean of $`X`$.
4. The ratio (last/this) of the second derivatives is close to 1:
    * $` 1 -`$ `scnd_der_ratio_range` $`< |\frac{x''_{k-1}}{x_{k''}}| < 1 + `$`scnd_der_ratio_range`
5. The ratio (this/next) of the second derivatives is sufficiently height:
    * $`|\frac{x''_{k}}{x''_{k+1}}| > `$`scnd_der_ratio_thresh`

NOTE:
- Only works for time series
- The time series is expected to be harmonized to an
  [equidistant frequency grid](docs/funcs/TimeSeriesHarmonization.md)


This Function is a generalization of the spectrum based spike flagging
mechanism as presented in [1].

### References
[1] Dorigo,W. et al.: Global Automated Quality Control of In Situ Soil Moisture
    Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
    doi:10.2136/vzj2012.0097.

