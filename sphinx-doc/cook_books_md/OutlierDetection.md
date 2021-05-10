# Basic Outlier Detection Workflows

## Data 

The [data set](https://git.ufz.de/rdm-software/saqc/-/blob/cookBux/sphinx-doc/ressources/data/incidentsLKG.csv) can be 
downloaded from the saqc git repository.

The data represents incidents of SARS-CoV-2 infections, on a daily basis, as reported by the 
[RKI](https://www.rki.de/DE/Home/homepage_node.html) in 2020. 

![](../ressources/images/cbooks_incidents1.png)

## Outlier

In June, an extreme spike can be observed. This spike relates to an incidence of so called "superspreading" in a local
[meat factory](https://www.heise.de/tp/features/Superspreader-bei-Toennies-identifiziert-4852400.html).
  
For the sake of modelling the spread of Covid, it can be of advantage, to filter the data for such extreme events, since
they may not be consistent with underlying distributional assumptions and thus interfere with the parameter learning 
process of the modelling.

To just introduce into some basic `SaQC` workflows, we will concentrate on classic variance based outlier detection approaches.

## Preparation
We, initially want to import the relevant packages. 

```python
import saqc
import pandas
import numpy as np
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt
``` 

We include the data via pandas [csv file parser](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html). 
This will give us a [data frame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) object, 
that we can directly feed into SaQC, in order to generate an SaQC object.

```python
import pandas as pd
# data path variable should point to where you have the incidents data set stored.
i_data = pd.read_csv(data_path)
i_data.index = pd.DatetimeIndex(i_data.index)
i_saqc = saqc.SaQC(data=i_data)
```

We can look at the data, via:

```python
saqc.show('incidents')
```

## Modelling

First, we want to model our data, to obtain a stationary, residuish variable with zero mean.
In SaQC, the results of data processing function, defaultly overrides the processed data column. 
So, if we want to transform our input data and reuse the original data later on, we need to duplicate 
it first, with the :py:func:`saqc.tools.copy <docs.func_modules.outliers.flagRange>` method:

```python
i_saqc = i_saqc.tools.copy(field='incidents', new_field='incidents_model')
```

The copy method has 2 parameters - the `field` parameter controlls the name of the variable to
copy, the `new_field` parameter holds the new column name of the duplicated variable. 

Easiest thing to do, would be, to apply some rolling mean
model via the :py:func:`saqc.rolling.roll <docs.func_modules.rolling.roll>` method.

```python
i_saqc = i_saqc.rolling.roll(field='incidents_model', func=np.mean, winsz='13D')
```

Then `winsz` parameter controlls the size of the rolling window. It can be fed any so called [date alias](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) string. We chose the rolling window to have a 13 days span.
You can pass arbitrary functions to the `func` parameter, to be applid to calculate every single windows "score". For example, you could go for the `median` instead of the `mean`. 

We calculate another model curve for the `"incidents"` data with the `np.mean` function from the `numpy` library. To not have to copy the original `incidents` variable everytime, we want to process it, we can make use of a shortcut by using the `target` parameter.

```python
i_saqc = i_saqc.rolling.roll(field='incidents', target='incidents_median', func=np.median, winsz='13D')
```
The `target` parameter can be passed to an function. It will determine the result of the function to be written to the data under the fieldname specified by it. If there already exists a field with the name passed to `target`, the data stored to this field will will be overridden.

Another common approach, is, to fit polynomials of certain degrees to the data. This could, of course, also be applied 
via a function passed to the rolling method - since this can get computationally expensive easily, for greater data sets, *SaQC* offers a build-in polynomial fit function 
:py:func:`saqc.curvefit.fitPolynomial <docs.func_modules.curvefit.fitPolynomial>`:

```python
i_saqc = i_saqc.curvefit.fitPolynomial(field='incidents', target='incidents_polynomial', polydeg=2 ,winsz='13D')
```

If you want to apply a completely arbitrary function to your data, without rolling, for example
a smoothing filter from the [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html) 
module, you would simply have to wrap the desired function up into a function of a single
array-like variable. To wrap the scipy butterworth filter for example, into a forward-backward application,
you would need to define a function first:

```python
def butterFilter(x, filter_order, nyq, cutoff, filter_type):
    b, a = butter(N=filter_order, Wn=cutoff / nyq, btype=filter_type)
    return filtfilt(b, a, x)
```

Than you can wrap it up with a lambda function, so it only has one free parameter and pass it to the 
:py:func:`saqc.transformation.transform <docs.func_modules.transformation.transform>` 
methods `func` argument.

```python
wrapped_func=lambda x: butterFilter(x, cutoff=0.1, nyq=0.5, filter_order=2)
i_saqc = i_saqc.tools.copy(field='incidents', new_field='incidents_lowPass')
i_saqc = i_saqc.transformation.transform(field='incidents_lowPass', wrapped_func=func)
```

You can check out the modelling results. Therefor we evaluate the qeued manipualations to the saqc object and return the results.

```python
i_saqc = i_saqc.evaluate()
result_data, _ saqc.getResult()
result_data.plot()
```

![](../ressources/images/cbooks_incidents2.png)

## Residues calculation

We want to evaluate the residues of the model, in order to score the outlierish-nes of every point. 
First, we retrieve the residues via the :py:func:`saqc.generic.process <docs.func_modules.generic.process>` method.
The method generates a new variable, resulting from the processing of other variables. It automatically
generates the field name it gets passed - so we do not have to generate new variable beforehand. The function we apply 
is just the computation of the variables difference for any timestep.

```python
i_saqc = i_saqc.generic.process('incidents_residues', func=lambda incidents, incidents_model:incidents - incidents_model)
```

Next, we score the residues simply by computing their [Z-scores](https://en.wikipedia.org/wiki/Standard_score).
The Z-score of a point $`x`$, relative to its surrounding $`D`$, evaluates to $`Z(x) = \frac{x - \mu(D)}{\sigma(D)}`$.

So, if we would like to roll with a window of a fixed size of 27 periods through the data and calculate the Z score for the point lying in the center of every window, we would define our function `z_score`:

```python
z_score = lambda D: abs((D[14] - np.mean(D)) / np.std(D)) 
```

And than do:

```python
i_saqc.rolling.roll(field='incidents_residues', target='incidents_scores', func=z_scores, winsz='13D')
```

The problem with this attempt, is, that it might get really slow for large data sets, because our function `z_scores` does not get decomposed into optimized building blocks - since it is a black box within `saqc`. Also, it relies on every window having a fixed number of values. otherwise, `D[14]` might not always be the value in the middle of the window, or it might not even exist, and an error will be thrown. 

If you want to accelerate your calculations and make them much more stable, it might be useful to decompose the scoring into seperate `rolling` calls. 

To make use of the fact, that `saqc`s rolling method trys to call optimized built-ins, and also, that it the return value of the rolling method is centered by default - we could calculate the series of the residues Mean and standard deviation seperately: 

```python
i_saqc = i_saqc.rolling.roll(field='incidents_residues', target='residues_mean', winsz='27D', 
                             func=np.mean)
i_saqc = i_saqc.rolling.roll(field='incidents_residues', target='residues_std', winsz='27D', 
                             func=np.std)
```
This will be noticably faster, since `saqc` dispatches the rolling with the basic numpy statistic methods to an optimized pandas built-in.
Also, as a result, all the values are centered and we dont have to care about window center indices, when we generate the *Z scores* form the series. 

```python
i_saqc = i_saqc.processGeneric(fields=['incidents_residues','incidents_mean','incidents_std'], target='incidents_scores', func=lambda x,y,z: abs((x-y) / z))
```

Lets evaluate the residues calculation and have a look at the resulting scores:
```python
i_saqc = i_saqc.evaluate()
i_saqc.show('incidents_scores')
```


## Setting Flag und unsetting Flags

We can now implement the common rule of thumb, that any Z-score value above 3, may indicate an outlierish data point, by:

```python
i_saqc = i_saqc.flagRange('incidents_scores', max=3).evaluate()
```

Now flags have been calculated for the scores:

```python
i_saqc.show('incidents_scores')
```

We now could project those flags onto our original incidents timeseries:

```python
i_saqc = i_saqc.flagGeneric(field=['incidents_scores'], target='incidents', func=lambda x: isFlagged(x))
```

Note, that we could have skipped the range flagging step, by including the lowpassing in our generic expression:

```python
i_saqc = i_saqc.flagGeneric(field=['incidents_scores'], target='incidents', func=lambda x: x > 3)
```

Lets check the result:

```python
i_saqc = i_saqc.evaluate
i_saqc.show('incidents')
```

Obveously, there are some flags set, that relate to minor incidents spikes relatively to there surrounding, but may not relate to global extreme values. Especially the left most flag seems not to relate to an extreme event at all. There is a lot of possibillities to tackle the issue. For example, we could try to impose the additional condition, that an outlier must relate to a sufficiently large residue. 

**TODO: (following doesnt work)**
```python
i_saqc.generic.flag(field='incidents','incidents_residues', target='incidents', func=lambda x,y: isflagged(x) & (y < 200), flag=-np.inf)
```

Note, that we could have skipped the unflagging step as well, by including the minimum condition for the residues in the initial generic expression as well, via:

```python
i_saqc = i_saqc.flagGeneric(field=['incidents_scores', 'incidents_residues'], target='incidents', func=lambda x, y: (x > 3) & (y < 200))
```
