.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. plot::
   :context: reset
   :include-source: False

   import matplotlib
   import saqc
   import pandas as pd

   data_path = '../resources/data/incidentsLKG.csv'
   data = pd.read_csv(data_path, index_col=0)
   data.index = pd.DatetimeIndex(data.index)

Outlier Detection and Flagging
==============================

The tutorial aims to introduce the usage of ``saqc`` methods in order to detect outliers in an uni-variate set up.
The tutorial guides through the following steps:


#. We checkout and load the example data set. Subsequently, we initialise an :py:class:`SaQC <saqc.core.core.SaQC>` object.

   * :ref:`Preparation <cookbooks/OutlierDetection:Preparation>`

     * :ref:`Data <cookbooks/OutlierDetection:Data>`
     * :ref:`Initialisation <cookbooks/OutlierDetection:Initialisation>`

#. We will see how to apply different smoothing methods and models to the data in order to obtain usefull residual
   variables.


   * :ref:`Modelling <cookbooks/OutlierDetection:Modelling>`

     * :ref:`Rolling Mean <cookbooks/OutlierDetection:Rolling Mean>`
     * :ref:`Rolling Median <cookbooks/OutlierDetection:Rolling Median>`
     * :ref:`Polynomial Fit <cookbooks/OutlierDetection:Polynomial Fit>`
     * :ref:`Custom Models <cookbooks/OutlierDetection:Custom Models>`

   * :ref:`Evaluation and Visualisation <cookbooks/OutlierDetection:Visualisation>`

#. We will see how we can obtain residuals and scores from the calculated model curves.


   * :ref:`Residuals and Scores <cookbooks/OutlierDetection:Residuals and Scores>`

     * :ref:`Residuals <cookbooks/OutlierDetection:Residuals>`
     * :ref:`Scores <cookbooks/OutlierDetection:Scores>`
     * :ref:`Optimization by Decomposition <cookbooks/OutlierDetection:Optimization by Decomposition>`

#. Finally, we will see how to derive flags from the scores itself and impose additional conditions, functioning as
   correctives.


   * :ref:`Setting and Correcting Flags <cookbooks/OutlierDetection:Setting and Correcting Flags>`

     * :ref:`Flagging the Scores <cookbooks/OutlierDetection:Flagging the Scores>`
     * `Additional Conditions ("unflagging") <#Additional-Conditions>`_
     * :ref:`Including Multiple Conditions <cookbooks/OutlierDetection:Including Multiple Conditions>`

Preparation
-----------

Data
^^^^

The example `data set <https://git.ufz.de/rdm-software/saqc/-/blob/cookBux/sphinx-doc/resources/data/incidentsLKG.csv>`_
is selected to be small, comprehendable and its single anomalous outlier
can be identified easily visually:

.. plot::
   :context:
   :include-source: False
   :width: 80 %
   :class: center

   data.plot()


It can be downloaded from the saqc git `repository <https://git.ufz.de/rdm-software/saqc/-/blob/cookBux/sphinx-doc/resources/data/incidentsLKG.csv>`_.

The data represents incidents of SARS-CoV-2 infections, on a daily basis, as reported by the
`RKI <https://www.rki.de/DE/Home/homepage_node.html>`_ in 2020.

In June, an extreme spike can be observed. This spike relates to an incidence of so called "superspreading" in a local
`meat factory <https://www.heise.de/tp/features/Superspreader-bei-Toennies-identifiziert-4852400.html>`_.

For the sake of modelling the spread of Covid, it can be of advantage, to filter the data for such extreme events, since
they may not be consistent with underlying distributional assumptions and thus interfere with the parameter learning
process of the modelling. Also it can help to learn about the conditions severely facilitating infection rates.

To introduce into some basic ``saqc`` workflows, we will concentrate on classic variance based outlier detection approaches.

Initialisation
^^^^^^^^^^^^^^

We initially want to import the data into our workspace. Therefore we import the `pandas <https://pandas.pydata.org/>`_
library and use its csv file parser `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_.


.. doctest:: exampleOD

   >>> data_path = './resources/data/incidentsLKG.csv'
   >>> import pandas as pd
   >>> data = pd.read_csv(data_path, index_col=0)


The resulting ``data`` variable is a pandas `data frame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
object. We can generate an :py:class:`SaQC <saqc.core.core.SaQC>` object directly from that. Beforehand we have to make sure, the index
of ``data`` is of the right type.

.. doctest:: exampleOD

   >>> data.index = pd.DatetimeIndex(data.index)

Now we do load the saqc package into the workspace and generate an instance of :py:class:`SaQC <saqc.core.core.SaQC>` object,
that refers to the loaded data.

.. plot::
   :context: close-figs
   :include-source: False

   import saqc
   qc = saqc.SaQC(data)

.. doctest:: exampleOD

   >>> import saqc
   >>> qc = saqc.SaQC(data)

The only timeseries have here, is the *incidents* dataset. We can have a look at the data and obtain the above plot through
the method :py:meth:`~saqc.SaQC.plot`:

.. doctest:: exampleOD

   >>> qc.plot('incidents') # doctest: +SKIP


Modelling
---------

First, we want to model our data in order to obtain a stationary, residuish variable with zero mean.

Rolling Mean
^^^^^^^^^^^^

Easiest thing to do, would be, to apply some rolling mean
model via the method :py:meth:`saqc.SaQC.roll`.

.. doctest:: exampleOD

   >>> import numpy as np
   >>> qc = qc.roll(field='incidents', target='incidents_mean', func=np.mean, window='13D')

.. plot::
   :context:
   :include-source: False

   import numpy as np
   qc = qc.roll(field='incidents', target='incidents_mean', func=np.mean, window='13D')

The ``field`` parameter is passed the variable name, we want to calculate the rolling mean of.
The ``target`` parameter holds the name, we want to store the results of the calculation to.
The ``window`` parameter controlls the size of the rolling window. It can be fed any so called `date alias <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ string. We chose the rolling window to have a 13 days span.

Rolling Median
^^^^^^^^^^^^^^

You can pass arbitrary function objects to the ``func`` parameter, to be applied to calculate every single windows "score".
For example, you could go for the *median* instead of the *mean*. The numpy library provides a `median <https://numpy.org/doc/stable/reference/generated/numpy.median.html>`_ function
under the name ``np.median``. We just calculate another model curve for the ``"incidents"`` data with the ``np.median`` function from the ``numpy`` library.

.. doctest:: exampleOD

   >>> qc = qc.roll(field='incidents', target='incidents_median', func=np.median, window='13D')

.. plot::
   :context:
   :include-source: False

   qc = qc.roll(field='incidents', target='incidents_median', func=np.median, window='13D')

We chose another :py:attr:`target` value for the rolling *median* calculation, in order to not override our results from
the previous rolling *mean* calculation.
The :py:attr:`target` parameter can be passed to any call of a function from the
saqc functions pool and will determine the result of the function to be written to the
data, under the fieldname specified by it. If there already exists a field with the name passed to ``target``\ ,
the data stored to this field will be overridden.

We will evaluate and visualize the different model curves :ref:`later <cookbooks/OutlierDetection:Visualisation>`.
Beforehand, we will generate some more model data.

Polynomial Fit
^^^^^^^^^^^^^^

Another common approach, is, to fit polynomials of certain degrees to the data.
:py:class:`SaQC <Core.Core.SaQC>` provides the polynomial fit function :py:meth:`~saqc.SaQC.fitPolynomial`:

.. doctest:: exampleOD

   >>> qc = qc.fitPolynomial(field='incidents', target='incidents_polynomial', order=2, window='13D')

.. plot::
   :context:
   :include-source: False

   qc = qc.fitPolynomial(field='incidents', target='incidents_polynomial', order=2, window='13D')


It also takes a :py:attr:`window` parameter, determining the size of the fitting window.
The parameter, :py:attr:`order` refers to the size of the rolling window, the polynomials get fitted to.

Custom Models
^^^^^^^^^^^^^

If you want to apply a completely arbitrary function to your data, without pre-chunking it by a rolling window,
you can make use of the more general :py:meth:`~saqc.SaQC.processGeneric` function.

Lets apply a smoothing filter from the `scipy.signal <https://docs.scipy.org/doc/scipy/reference/signal.html>`_
module. We wrap the filter generator up into a function first:

.. testcode:: exampleOD

   from scipy.signal import filtfilt, butter
   def butterFilter(x, filter_order, nyq, cutoff, filter_type="lowpass"):
       b, a = butter(N=filter_order, Wn=cutoff / nyq, btype=filter_type)
       return pd.Series(filtfilt(b, a, x), index=x.index)

.. plot::
   :context:
   :include-source: False

   from scipy.signal import filtfilt, butter
   def butterFilter(x, filter_order, nyq, cutoff, filter_type="lowpass"):
       b, a = butter(N=filter_order, Wn=cutoff / nyq, btype=filter_type)
       return pd.Series(filtfilt(b, a, x), index=x.index)


This function object, we can pass on to the :py:meth:`~saqc.SaQC.processGeneric` methods ``func`` argument.

.. doctest:: exampleOD

   >>> qc = qc.processGeneric(field='incidents', target='incidents_lowPass',
   ... func=lambda x: butterFilter(x, cutoff=0.1, nyq=0.5, filter_order=2))

.. plot::
   :context:
   :include-source: False

   qc = qc.processGeneric(field='incidents', target='incidents_lowPass', func=lambda x: butterFilter(x, cutoff=0.1, nyq=0.5, filter_order=2))


Visualisation
-------------

We can obtain those updated informations by generating a `pandas dataframe <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
representation of it, with the :py:attr:`data <saqc.core.core.SaQC.data>` method:

.. doctest:: exampleOD

   >>> data = qc.data

.. plot::
   :context:
   :include-source: False

   data = qc.data

To see all the results obtained so far, plotted in one figure window, we make use of the dataframes `plot <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html>`_ method.

.. doctest:: exampleOD

   >>> data.to_df().plot()
   <AxesSubplot:>

.. plot::
   :context:
   :include-source: False
   :width: 80 %
   :class: center

   data.to_df().plot()


Residuals and Scores
--------------------

Residuals
^^^^^^^^^

We want to evaluate the residuals of one of our models model, in order to score the outlierish-nes of every point.
Therefor we just stick to the initially calculated rolling mean curve.

First, we retrieve the residuals via the :py:meth:`~saqc.SaQC.processGeneric` method.
This method always comes into play, when we want to obtain variables, resulting from basic algebraic
manipulations of one or more input variables.

For obtaining the models residuals, we just subtract the model data from the original data and assign the result
of this operation to a new variable, called ``incidents_residuals``. This Assignment, we, as usual,
control via the ``target`` parameter.

.. doctest:: exampleOD

   >>> qc = qc.processGeneric(['incidents', 'incidents_mean'], target='incidents_residuals', func=lambda x, y: x - y)

.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.processGeneric(['incidents', 'incidents_mean'], target='incidents_residuals', func=lambda x, y: x - y)


Scores
^^^^^^

Next, we score the residuals simply by computing their `Z-scores <https://en.wikipedia.org/wiki/Standard_score>`_.
The *Z*-score of a point :math:`x`, relative to its surrounding :math:`D`,
evaluates to :math:`Z(x) = \frac{x - \mu(D)}{\sigma(D)}`.

So, if we would like to roll with a window of a fixed size of *27* periods through the data and calculate the *Z*\ -score
for the point lying in the center of every window, we would define our function ``z_score``\ :

.. doctest:: exampleOD

   >>> z_score = lambda D: abs((D[14] - np.mean(D)) / np.std(D))

.. plot::
   :context: close-figs
   :include-source: False

   z_score = lambda D: abs((D[14] - np.mean(D)) / np.std(D))

And subsequently, use the :py:meth:`~saqc.SaQC.roll` method to make a rolling window application with the scoring
function:

.. doctest:: exampleOD

   >>> qc = qc.roll(field='incidents_residuals', target='incidents_scores', func=z_score, window='27D')

.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.roll(field='incidents_residuals', target='incidents_scores', func=z_score, window='27D')

Optimization by Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are 2 problems with the attempt presented :ref:`above <cookbooks/OutlierDetection:Scores>`.

First, the rolling application of the customly
defined function, might get really slow for large data sets, because our function ``z_scores`` does not get decomposed into optimized building blocks.

Second, and maybe more important, it relies heavily on every window having a fixed number of values and a fixed temporal extension.
Otherwise, ``D[14]`` might not always be the value in the middle of the window, or it might not even exist,
and an error will be thrown.

So the attempt works fine, only because our data set is small and strictly regularily sampled.
Meaning that it has constant temporal distances between subsequent meassurements.

In order to tweak our calculations and make them much more stable, it might be useful to decompose the scoring
into seperate calls to the :py:meth:`~saqc.SaQC.roll` function, by calculating the series of the
residuals *mean* and *standard deviation* seperately:

.. doctest:: exampleOD

   >>> qc = qc.roll(field='incidents_residuals', target='residuals_mean', window='27D', func=np.mean)
   >>> qc = qc.roll(field='incidents_residuals', target='residuals_std', window='27D', func=np.std)
   >>> qc = qc.processGeneric(field=['incidents_scores', "residuals_mean", "residuals_std"], target="residuals_norm",
   ... func=lambda this, mean, std: (this - mean) / std)


.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.roll(field='incidents_residuals', target='residuals_mean', window='27D', func=np.mean)
   qc = qc.roll(field='incidents_residuals', target='residuals_std', window='27D', func=np.std)
   qc = qc.processGeneric(field=['incidents_scores', "residuals_mean", "residuals_std"], target="residuals_norm", func=lambda this, mean, std: (this - mean) / std)


With huge datasets, this will be noticably faster, compared to the method presented :ref:`initially <cookbooks/OutlierDetection:Scores>`\ ,
because ``saqc`` dispatches the rolling with the basic numpy statistic methods to an optimized pandas built-in.

Also, as a result of the :py:meth:`~saqc.SaQC.roll` assigning its results to the center of every window,
all the values are centered and we dont have to care about window center indices when we are generating
the *Z*\ -Scores from the two series.

We simply combine them via the
:py:meth:`~saqc.SaQC.processGeneric` method, in order to obtain the scores:

.. doctest:: exampleOD

   >>> qc = qc.processGeneric(field=['incidents_residuals','residuals_mean','residuals_std'],
   ... target='incidents_scores', func=lambda x,y,z: abs((x-y) / z))

.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.processGeneric(field=['incidents_residuals','residuals_mean','residuals_std'], target='incidents_scores', func=lambda x,y,z: abs((x-y) / z))



Let's have a look at the resulting scores:

.. doctest:: exampleOD

   >>> qc.plot('incidents_scores') # doctest:+SKIP


.. plot::
   :context: close-figs
   :include-source: False
   :width: 80 %
   :class: center

   qc.plot('incidents_scores')


Setting and correcting Flags
----------------------------

Flagging the Scores
^^^^^^^^^^^^^^^^^^^

We can now implement the common `rule of thumb <https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule>`_\ ,
that any *Z*\ -score value above *3* may indicate an outlierish data point,
by applying the :py:meth:`~saqc.SaQC.flagRange` method with a `max` value of *3*.

.. doctest:: exampleOD

   >>> qc = qc.flagRange('incidents_scores', max=3)

.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.flagRange('incidents_scores', max=3)

Now flags have been calculated for the scores:

>>> qc.plot('incidents_scores') # doctest:+SKIP


.. plot::
   :context: close-figs
   :include-source: False
   :width: 80 %
   :class: center

   qc.plot('incidents_scores')


Projecting Flags
^^^^^^^^^^^^^^^^

We now can project those flags onto our original incidents timeseries:

.. doctest:: exampleOD

   >>> qc = qc.flagGeneric(field=['incidents_scores'], target='incidents', func=lambda x: isflagged(x))

.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.flagGeneric(field=['incidents_scores'], target='incidents', func=lambda x: isflagged(x))

Note, that we could have skipped the :ref:`range flagging step <cookbooks/OutlierDetection:Flagging the scores>`\ , by including the cutting off in our

generic expression:

.. doctest:: exampleOD

   >>> qc = qc.flagGeneric(field=['incidents_scores'], target='incidents', func=lambda x: x > 3)

Lets check out the results:

.. doctest:: exampleOD

   >>> qc.plot('incidents') # doctest: +SKIP

.. plot::
   :context: close-figs
   :include-source: False
   :width: 80 %
   :class: center

   qc.plot('incidents')


Obveously, there are some flags set, that, relative to their 13 days surrounding, might relate to minor incidents spikes,
but may not relate to superspreading events we are looking for.

Especially the left most flag seems not to relate to an extreme event at all.
This overflagging stems from those values having a surrounding with very low data variance, and thus, evaluate to a relatively high Z-score.

There are a lot of possibilities to tackle the issue. In the next section, we will see how we can improve the flagging results
by incorporating additional domain knowledge.

Additional Conditions
---------------------

In order to improve our flagging result, we could additionally assume, that the events we are interested in,
are those with an incidents count that is deviating by a margin of more than
*20* from the 2 week average.

This is equivalent to imposing the additional condition, that an outlier must relate to a sufficiently large residual.

Unflagging
^^^^^^^^^^

We can do that posterior to the preceeding flagging step, by *removing*
some flags based on some condition.

In order want to *unflag* those values, that do not relate to
sufficiently large residuals, we assign them the :py:const:`~saqc.constants.UNFLAGGED` flag.

Therefore, we make use of the :py:meth:`~saqc.SaQC.flagGeneric` method.
This method usually comes into play, when we want to assign flags based on the evaluation of logical expressions.

So, we check out, which residuals evaluate to a level below *20*\ , and assign the
flag value for :py:const:`~saqc.constants.UNFLAGGED`. This value defaults to
to ``-np.inf`` in the default translation scheme, wich we selected implicitly by not specifying any special scheme in the
generation of the :py:class:`~Core.Core.SaQC>` object in the :ref:`beginning <cookbooks/OutlierDetection:Initialisation>`.

.. doctest:: exampleOD

   >>> qc = qc.flagGeneric(field=['incidents','incidents_residuals'], target="incidents",
   ... func=lambda x,y: isflagged(x) & (y < 50), flag=-np.inf)


.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.flagGeneric(field=['incidents','incidents_residuals'], target="incidents", func=lambda x,y: isflagged(x) & (y < 50), flag=-np.inf)


Notice, that we passed the desired flag level to the :py:attr:`flag` keyword in order to perform an
"unflagging" instead of the usual flagging. The :py:attr:`flag` keyword can be passed to all the functions
and defaults to the selected translation schemes :py:const:`BAD <saqc.constants.BAD>` flag level.

Plotting proofs the tweaking did in deed improve the flagging result:

.. doctest:: exampleOD

   >>> qc.plot("incidents") # doctest:+SKIP


.. plot::
   :context: close-figs
   :include-source: False
   :width: 80 %
   :class: center

   qc.plot("incidents")


Including multiple conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we do not want to first set flags, only to remove the majority of them in the next step, we also
could circumvent the :ref:`unflagging <cookbooks/OutlierDetection:Unflagging>` step, by adding to the call to
:py:meth:`~saqc.SaQC.flagRange` the condition for the residuals having to be above *20*

.. doctest:: exampleOD

   >>> qc = qc.flagGeneric(field=['incidents_scores', 'incidents_residuals'], target='incidents',
   ... func=lambda x, y: (x > 3) & (y > 20))
   >>> qc.plot("incidents") # doctest: +SKIP

.. plot::
   :context: close-figs
   :include-source: False
   :width: 80 %
   :class: center

   qc = qc.flagGeneric(field=['incidents_scores', 'incidents_residuals'], target='incidents', func=lambda x, y: (x > 3) & (y > 20))
   qc.plot("incidents")
