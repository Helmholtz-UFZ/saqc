# Spike Detection

A collection of quality check routines to find spikes.

## Index

- [spikes_basic](#spikes_basic)
- [spikes_simpleMad](#spikes_simplemad)
- [spikes_slidingZscore](#spikes_slidingzscore)
- [spikes_spektrumBased](#spikes_spektrumbased)


## spikes_basic

```
spikes_basic(thresh, tolerance, window_size)
```

| parameter | data type                                                     | default value | description                                                                                    |
|-----------|---------------------------------------------------------------|---------------|------------------------------------------------------------------------------------------------|
| thresh    | float                                                         |               | Minimum difference between to values, to consider the latter one as a spike. See condition (1) |
| tolerance | float                                                         |               | Maximum difference between pre-spike and post-spike values. See condition (2)                  |
| window    | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | Maximum length of "spiky" value courses. See condition (3)                                  |

A basic outlier test, that is designed to work for harmonized, as well as raw
(not-harmonized) data.

The values $`x_{n}, x_{n+1}, .... , x_{n+k} `$ of a time series $`x_t`$ with 
timestamps $`t_i`$ are considered spikes, if:

1. $`|x_{n-1} - x_{n+s}| > `$ `thresh`, $` s \in \{0,1,2,...,k\} `$

2. $`|x_{n-1} - x_{n+k+1}| < `$ `tolerance`

3. $` |t_{n-1} - t_{n+k+1}| < `$ `window`

By this definition, spikes are values, that, after a jump of margin `thresh`(1),
are keeping that new value level, for a time span smaller than
`window` (3), and then return to the initial value level -
within a tolerance of `tolerance` (2).

NOTE:
This characterization of a "spike", not only includes one-value
outliers, but also plateau-ish value courses.


## spikes_simpleMad

```
spikes_simpleMad(window="1h", z=3.5)
```

| parameter | data type                                                             | default value | description                                                          |
|-----------|-----------------------------------------------------------------------|---------------|----------------------------------------------------------------------|
| window    | integer/[offset string](docs/ParameterDescriptions.md#offset-strings) | `"1h"`        | size of the sliding window, where the modified Z-score is applied on |
| z         | float                                                                 | `3.5`         | z-parameter of the modified Z-score                                  |

This functions flags outliers using the simple median absolute deviation test.

Values are flagged if they fulfill the following condition within a sliding window:

```math
 0.6745 * |x - m| > mad * z > 0
```

where $`x`$ denotes the window data, $`m`$ the window median, $`mad`$ the median
absolute deviation and $`z`$ the $`z`$-parameter of the modified Z-Score.

The window is moved by one time stamp at a time.

NOTE:
This function should only be applied on normalized data.

References:
[1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm


## spikes_slidingZscore

```
spikes_slidingZscore(window="1h", offset="1h", count=1, polydeg=1, z=3.5, method="modZ")
```

| parameter | data type                                                             | default value | description                                                 |
|-----------|-----------------------------------------------------------------------|---------------|-------------------------------------------------------------|
| window    | integer/[offset string](docs/ParameterDescriptions.md#offset-strings) | `"1h"`        | size of the sliding window                                  |
| offset    | integer/[offset string](docs/ParameterDescriptions.md#offset-strings) | `"1h"`        | offset between two consecutive windows                      |
| count     | integer                                                               | `1`           | the minimal count a possible outlier needs, to be flagged   |
| polydeg   | integer                                                               | `1"`          | the degree of the polynomial fit, to calculate the residual |
| z         | float                                                                 | `3.5`         | z-parameter for the *method* (see description)              |
| method    | [string](#outlier-detection-methods)                                  | `"modZ"`      | the method to detect outliers                               |

This functions flags spikes using the given method within sliding windows.

NOTE:
 - `window` and `offset` must be of same type, mixing of offset- and integer-
    based windows is not supported and will fail
 - offset-strings only work with time-series-like data

The algorithm works as follows:
  1.  a window of size `window` is cut from the data
  2.  normalization - the data is fit by a polynomial of the given degree `polydeg`, which is subtracted from the data
  3.  the outlier detection `method` is applied on the residual, possible outlier are marked
  4.  the window (on the data) is moved by `offset`
  5.  start over from 1. until the end of data is reached
  6.  all potential outliers, that are detected `count`-many times, are flagged as outlier

### Outlier Detection Methods
Currently two outlier detection methods are implemented:

1. `"zscore"`: The Z-score marks every value as a possible outlier, which fulfills the following condition:

   ```math
    |r - m| > s * z
   ```
   where $`r`$ denotes the residual, $`m`$ the residual mean, $`s`$ the residual
   standard deviation, and $`z`$ the $`z`$-parameter.

2. `"modZ"`: The modified Z-score Marks every value as a possible outlier, which fulfills the following condition:

   ```math
    0.6745 * |r - m| > mad * z > 0
   ```

   where $`r`$ denotes the residual, $`m`$ the residual mean, $`mad`$ the residual median absolute
   deviation, and $`z`$ the $`z`$-parameter.

### References
[1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm


## spikes_spektrumBased

```
spikes_spektrumBased(raise_factor=0.15, deriv_factor=0.2,
                     noise_thresh=1, noise_window="12h", noise_func="CoVar",
                     smooth_poly_deg=2, smooth_window=None)
```

| parameter       | data type                                                     | default value | description                                                                                                                                                |
|-----------------|---------------------------------------------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| raise_factor    | float                                                         | `0.15`        | Minimum relative value difference between two values to consider the latter as a spike candidate. See condition (1)                                        |
| deriv_factor    | float                                                         | `0.2`         | See condition (2)                                                                                                                                          |
| noise_thresh    | float                                                         | `1`           | Upper threshold for noisiness of data surrounding potential spikes. See condition (3)                                                                      |
| noise_window    | [offset string](docs/ParameterDescriptions.md#offset-strings) | `"12h"`       | Determines the range of the time window of the "surrounding" data of a potential spike. See condition (3)                                                  |
| noise_func      | [string](#noise-detection-functions)                          | `"CoVar"`     | Function to calculate noisiness of data, surrounding potential spikes                                                                                      |
| smooth_poly_deg | integer                                                       | `2`           | Degree of the polynomial used for fitting with the Savitsky-Golay filter                                                                                   |
| smooth_window   | [offset string](docs/ParameterDescriptions.md#offset-strings) | `None`        | Size of the smoothing window of the Savitsky-Golay filter. The default value `None` results in a window of two times the sampling rate (i.e. three values) |


The function flags spikes by evaluating the time series' derivatives
and applying various conditions to them.

The value $`x_{k}`$ of a time series $`x_t`$ with 
timestamps $`t_i`$ is considered a spikes, if:


1. The quotient to its preceding data point exceeds a certain bound:
    * $` |\frac{x_k}{x_{k-1}}| > 1 + `$ `raise_factor`, or
    * $` |\frac{x_k}{x_{k-1}}| < 1 - `$ `raise_factor`
2. The quotient of the second derivative $`x''`$, at the preceding
   and subsequent timestamps is close enough to 1:
    * $` |\frac{x''_{k-1}}{x''_{k+1}} | > 1 - `$ `deriv_factor`, and
    * $` |\frac{x''_{k-1}}{x''_{k+1}} | < 1 + `$ `deriv_factor`
3. The dataset $`X = x_i, ..., x_{k-1}, x_{k+1}, ..., x_j`$, with 
   $`|t_{k-1} - t_i| = |t_j - t_{k+1}| =`$ `noise_window` fulfills the 
   following condition: 
   `noise_func`$`(X) <`$ `noise_thresh`
   
NOTE:
- The dataset is supposed to be harmonized to a time series with an equidistant frequency grid
- The derivative is calculated after applying a Savitsky-Golay filter to $`x`$

  This function is a generalization of the Spectrum based Spike flagging
  mechanism presented in [1]

### Noise Detection Functions
Currently two different noise detection functions are implemented:
- `"CoVar"`: Coefficient of Variation
- `"rVar"`: relative Variance


### References
[1] Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
    Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
    doi:10.2136/vzj2012.0097.
