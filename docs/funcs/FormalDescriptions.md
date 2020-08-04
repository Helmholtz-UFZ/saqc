# Mathematical descriptions

A collection of detailed mathematical descriptions.

## Index

- [spikes_flagRaise](#spikes_flagraise)
- [spikes_flagSpektrumBased](#spikes_flagspektrumbased)
- [breaks_flagSpektrumBased](#breaks_flagspektrumbased)


## spikes_flagRaise

The value $`x_{k}`$ of a time series $`x`$ with associated 
timestamps $`t_i`$, is flagged a rise, if:

1. There is any value $`x_{s}`$, preceeding $`x_{k}`$ within `raise_window` range, so that:
    * $` M = |x_k - x_s | > `$  `thresh` $` > 0`$ 
2. The weighted average $`\mu^*`$ of the values, preceeding $`x_{k}`$ within `average_window` range indicates, that $`x_{k}`$ doesnt return from an outliererish value course, meaning that:  
    * $` x_k > \mu^* + ( M `$ / `mean_raise_factor` $`)`$  
3. Additionally, if `min_slope` is not `None`, $`x_{k}`$ is checked for being sufficiently divergent from its very predecessor $`x_{k-1}`$, meaning that, it is additionally checked if: 
    * $`x_k - x_{k-1} > `$ `min_slope` 
    * $`t_k - t_{k-1} > `$ `min_slope_weight`*`intended_freq`

The weighted average $`\mu^*`$ was calculated with weights $`w_{i}`$, defined by: 
* $`w_{i} = (t_i - t_{i-1})`$ / `intended_freq`, if $`(t_i - t_{i-1})`$ < `intended_freq` and $`w_i =1`$ otherwise. 



The value $`x_{k}`$ of a time series $`x_t`$ with 
timestamps $`t_i`$ is considered a spikes, if:


## spikes_flagSpektrumBased


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

## breaks_flagSpektrumBased

A value $`x_k`$ of a time series $`x_t`$ with timestamps $`t_i`$, is considered to be a break, if:

1. $`x_k`$ represents a sufficiently large relative jump:

   $`|\frac{x_k - x_{k-1}}{x_k}| >`$ `thresh_rel`

2. $`x_k`$ represents a sufficient absolute jump:

   $`|x_k - x_{k-1}| >`$ `thresh_abs`

3. The dataset $`X = x_i, ..., x_{k-1}, x_{k+1}, ..., x_j`$, with $`|t_{k-1} - t_i| = |t_j - t_{k+1}| =`$ `first_der_window`
   fulfills the following condition:
   
   $`|x'_k| >`$ `first_der_factor` $` \cdot \bar{X} `$
   
   where $`\bar{X}`$ denotes the arithmetic mean of $`X`$.

4. The ratio (last/this) of the second derivatives is close to 1:

   $` 1 -`$ `scnd_der_ratio_margin_1` $`< |\frac{x''_{k-1}}{x_{k''}}| < 1 + `$`scnd_der_ratio_margin_1`

5. The ratio (this/next) of the second derivatives is sufficiently height:

   $`|\frac{x''_{k}}{x''_{k+1}}| > `$`scnd_der_ratio_margin_2`