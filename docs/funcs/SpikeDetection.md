# Spike Detection

A collection of quality check routines to find spike.

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
| thresh    | float                                                         |               | Minimum difference between to values, to consider the latter one as a spike, see condition (1) |
| tolerance | float                                                         |               | Maximum difference between pre-spike and post-spike values Range of area, see condition (2)    |
| window    | [offset string](docs/ParameterDescriptions.md#offset-strings) |               | Maximum length of "spikish" value courses, see condition (3)                                   |

A basic outlier test, that is designed to work for harmonized, as well as raw
(not-harmonized) data.

The values $`x_{n}, x_{n+1}, .... , x_{n+k} `$ of a passed timeseries $`x`$,
are considered spikes, if:

1. $`|x_{n-1} - x_{n + s}| > `$ `thresh`, $` s \in \{0,1,2,...,k\} `$

2. $`|x_{n-1} - x_{n+k+1}| < `$ `tolerance`

3. $` |y_{n-1} - y_{n+k+1}| < `$ `window`, with $`y `$, denoting the series
   of timestamps associated with $`x `$.

By this definition, spikes are values, that, after a jump of margin `thresh`(1),
are keeping that new value level, for a timespan smaller than
`window` (3), and then return to the initial value level -
within a tolerance of `tolerance` (2).  

Note, that this characterization of a "spike", not only includes one-value
outliers, but also plateau-ish value courses.


## spikes_simpleMad

```
spikes_simpleMad(window="1h", z=3.5)
```

| parameter | data type                                                             | default value | description                                                          |
|-----------|-----------------------------------------------------------------------|---------------|----------------------------------------------------------------------|
| window    | integer/[offset string](docs/ParameterDescriptions.md#offset-strings) | `"1h"`        | size of the sliding window, where the modified Z-score is applied on |
| z         | float                                                                 | `3.5`         | z-parameter of the modified Z-score                                  |

This functions flags outliers by simple median absolute deviation test.
The *modified Z-score* [1] is used to detect outliers. Values are flagged if
they fulfill the following condition within a sliding window:

```math
 0.6745 * |x - m| > mad * z > 0
```

where $`x`$ denotes the window data, $`m`$ the window median, $`mad`$ the median absolute deviation and $`z`$ the
$`z`$-parameter of the modified Z-Score.

The window is moved by one time stamp at a time.

Note: This function should only be applied on normalized data.
 
See also:
[1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm


## spikes_slidingZscore

```
spikes_slidingZscore(window="1h", offset="1h", count=1, deg=1, z=3.5, method="modZ")
```

| parameter | data type                                                             | default value | description                                                 |
|-----------|-----------------------------------------------------------------------|---------------|-------------------------------------------------------------|
| window    | integer/[offset string](docs/ParameterDescriptions.md#offset-strings) | `"1h"`        | size of the sliding window                                  |
| offset    | integer/[offset string](docs/ParameterDescriptions.md#offset-strings) | `"1h"`        | offset between two consecutive windows                      |
| count     | integer                                                               | `1`           | the minimal count, a possible outlier needs, to be flagged  |
| deg       | integer                                                               | `1"`          | the degree of the polynomial fit, to calculate the residual |
| z         | float                                                                 | `3.5`         | z-parameter for the *method* (see description)              |
| method    | string                                                                | `"modZ"`      | the method outlier are detected with                        |

Detect outlier/spikes using a given method within sliding windows.

NOTE:
 - `window` and `offset` must be of same type, mixing of offset and integer is not supported and will fail
 - offset-strings only work with time-series-like data

The algorithm works as follows:
  1.  a window of size `window` is cut from the data
  2.  normalization - the data is fit by a polynomial of the given degree `deg`, which is subtracted from the data
  3.  the outlier detection `method` is applied on the residual, possible outlier are marked
  4.  the window (on the data) is moved by `offset`
  5.  start over from 1. until the end of data is reached
  6.  all potential outliers, that are detected `count`-many times, are flagged as outlier 

The possible outlier detection methods are *zscore* and *modZ*. 
In the following description, the residual (calculated from a slice by the sliding window)
is referred as *data*.

The **zscore** (Z-score) [1] marks every value as possible outlier, which fulfill:

```math
 |r - m| > s * z
```
where $`r`$ denotes the residual, $`m`$ the residual mean, $`s`$ the residual standard deviation,
and $`z`$ the $`z`$-parameter.

The **modZ** (modified Z-score) [1] marks every value as possible outlier, which fulfill:

```math
 0.6745 * |r - m| > mad * z > 0
```
where $`r`$ denotes the residual, $`m`$ the residual mean, $`mad`$ the residual median absolute
deviation, and $`z`$ the $`z`$-parameter.

See also:
[1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm


## spikes_spektrumBased


```
spikes_spektrumBased(raise_factor=0.15, dev_cont_factor=0.2,
                     noise_barrier=1, noise_window_size="12h", noise_statistic="CoVar",
                     smooth_poly_order=2, filter_window_size=None)
```

| parameter         | data type                                                     | default value | description                                                                                                                                                                                                                  |
|-------------------|---------------------------------------------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| raise_factor      | float                                                         | `0.15`        | Minimum margin of value change, a datapoint has to represent, to become a candidate for a spike. See condition (1).                                                                                                          |
| dev_cont_factor   | float                                                         | `0.2`         | See condition (2).                                                                                                                                                                                                           |
| noise_barrier     | float                                                         | `1`           | Upper bound for noisyness of data surrounding potential spikes. See condition (3).                                                                                                                                           |
| noise_window      | [offset string](docs/ParameterDescriptions.md#offset-strings) | `"12h"`       | offset string. Determines the range of the time window of the "surrounding" data of a potential spike. See condition (3).                                                                                                    |
| noise_statistic   | string                                                        | `"CoVar"`     | Operator to calculate noisyness of data, surrounding potential spikes. Either `"Covar"` (=Coefficient of Variation) or `"rvar"` (=relative Variance).                                                                        |
| smooth_poly_order | integer                                                       | `2`           | Order of the polynomial fit, applied with Savitsky-Golay-filter.                                                                                                                                                             |
| filter_window     | [offset string](docs/ParameterDescriptions.md#offset-strings) | `None`        | Controls the range of the smoothing window applied with the Savitsky-Golay filter. If `None` (default), the window size will be two times the sampling rate. (Thus, covering 3 values.) If unsure, do not change that value. |


The function detects and flags spikes in input data series by evaluating the
timeseries' derivatives and applying some conditions to them.

A datapoint $`x_k`$ of a dataseries $`x`$,
is considered a spike, if:

1. The quotient to its preceeding datapoint exceeds a certain bound:
    * $` |\frac{x_k}{x_{k-1}}| > 1 + `$ `raise_factor`, or
    * $` |\frac{x_k}{x_{k-1}}| < 1 - `$ `raise_factor`
2. The quotient of the data's second derivative $`x''`$, at the preceeding
   and subsequent timestamps is close enough to 1:
    * $` |\frac{x''_{k-1}}{x''_{k+1}} | > 1 - `$ `dev_cont_factor`, and
    * $` |\frac{x''_{k-1}}{x''_{k+1}} | < 1 + `$ `dev_cont_factor`   
3. The dataset, $`X_k`$, surrounding $`x_{k}`$, within `noise_window` range,
   but excluding $`x_{k}`$, is not too noisy. Whereas the noisyness gets measured
   by `noise_statistic`:
    * `noise_statistic`$`(X_k) <`$ `noise_barrier`

NOTE:
- The dataset is supposed to be harmonized to a timeseries with an equidistant frequency grid
- The derivative is calculated after applying a Savitsky-Golay filter to $`x`$


This function is a generalization of the Spectrum based Spike flagging
mechanism presented in:

Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
doi:10.2136/vzj2012.0097.

