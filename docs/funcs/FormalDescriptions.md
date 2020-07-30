# Mathematical descriptions

A collection of detailed mathematical descriptions.

## Index

- [spikes_flagRaise](#spikes_flagraise)


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