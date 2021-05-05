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

## Modelling

First, we want to model our data, to obtain a stationary, residuish variable with zero mean.
In SaQC, the results of data processing functions, defaultly overrides the processed data column. 
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

```python
i_saqc = i_saqc.rolling.roll(field='incidents_residues', target='residues_mean', winsz='27D', 
                             func=np.mean)
i_saqc = i_saqc.rolling.roll(field='incidents_residues', target='residues_std', winsz='27D', 
                             func=np.std)
i_saqc = i_saqc.generic.process(field='incidents_scores', 
                                func=lambda This, residues_mean, residues_std: (This - residues_mean)/residues_std )
```





