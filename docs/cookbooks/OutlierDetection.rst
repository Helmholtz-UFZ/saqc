.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later


Outlier Detection
=================

The tutorial aims to introduce into a simple to use, jet powerful method for clearing :ref:`uniformly sampled <cookbooks/DataRegularisation:Data Regularization>`, *univariate*
data, from global und local outliers as well as outlier clusters.
Therefor, we will introduce into the usage of the :py:meth:`~saqc.SaQC.flagUniLOF` method, which represents a
modification of the established `Local Outlier Factor <https://de.wikipedia.org/wiki/Local_Outlier_Factor>`_ (LOF)
algorithm and is applicable without prior modelling of the data to flag.

* :ref:`Example Data Import <cookbooks/OutlierDetection:Example Data Import>`
* :ref:`Initial Flagging <cookbooks/OutlierDetection:Initial Flagging>`
* :ref:`Tuning Threshold Parameter <cookbooks/OutlierDetection:Tuning Threshold Parameter>`
* :ref:`Tuning Locality Parameter <cookbooks/OutlierDetection:Tuning Locality Parameter>`


Example Data Import
-------------------

.. plot::
   :context: reset
   :include-source: False

   import matplotlib
   import saqc
   import pandas as pd
   data = pd.read_csv('../resources/data/hydro_data.csv')
   data = data.set_index('Timestamp')
   data.index = pd.DatetimeIndex(data.index)
   qc = saqc.SaQC(data)

We load the example `data set <https://git.ufz.de/rdm-software/saqc/-/blob/develop/docs/resources/data/hydro_data.csv>`_
from the *saqc* repository using the `pandas <https://pandas.pydata.org/>`_ csv
file reader.
Subsequently, we cast the index of the imported data to `DatetimeIndex <https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html>`, then initialize
a :py:class:`~saqc.SaQC` instance using the imported data and finally we plot
it via the built-in :py:meth:`~saqc.SaQC.plot` method.

.. doctest:: flagUniLOFExample

   >>> import saqc
   >>> data = pd.read_csv('./resources/data/hydro_data.csv')
   >>> data = data.set_index('Timestamp')
   >>> data.index = pd.DatetimeIndex(data.index)
   >>> qc = saqc.SaQC(data)
   >>> qc.plot('sac254_raw') # doctest: +SKIP

.. plot::
   :context:
   :include-source: False
   :class: center

    qc.plot('sac254_raw')

Initial Flagging
----------------

We start by applying the algorithm :py:meth:`~saqc.SaQC.flagUniLOF` with
default arguments, so the main calibration
parameters :py:attr:`n` and :py:attr:`thresh` are set to `20` and `1.5`
respectively.

For an detailed overview over all the parameters, as well as an introduction
into the working of the algorithm, see the documentation of :py:meth:`~saqc.SaQC.flagUniLOF`
itself.

.. doctest:: flagUniLOFExample

   >>> import saqc
   >>> qc = qc.flagUniLOF('sac254_raw')
   >>> qc.plot('sac254_raw') # doctest: +SKIP

.. plot::
   :context: close-figs
   :include-source: False
   :class: center
   :caption: Flagging result with default parameter configuration.

   qc = qc.flagUniLOF('sac254_raw')
   qc.plot('sac254_raw')

The results from that initial shot seem to look not too bad.
Most instances of obvious outliers seem to have been flagged right
away and there seem to be no instances of inliers having been falsely labeled.
Zooming in onto a 3 months strip on *2016*, gives the impression of
some not so extreme outliers having passed :py:meth:`~saqc.SaQC.flagUniLOF`
undetected:

.. plot::
   :context: close-figs
   :include-source: False
   :class: centers
   :caption: Assuming the flickering values in late september also qualify as outliers, we will see how to tune the algorithm to detect those in the next section.

   qc.plot('sac254_raw', xscope=slice('2016-09','2016-11'))

Tuning Threshold Parameter
--------------------------

Of course, the result from applying :py:meth:`~saqc.SaQC.flagUniLOF` with
default parameter settings might not always meet the expectations.

The best way to tune the algorithm, is, by tweaking one of the
parameters :py:attr:`thresh` or :py:attr:`n`.

To tune :py:attr:`thresh`, find a value that slightly underflags the data,
and *reapply* the function with evermore decreased values of
:py:attr:`thresh`.

.. doctest:: flagUniLOFExample

   >>> qc = qc.flagUniLOF('sac254_raw', thresh=1.3, label='threshold = 1.3')
   >>> qc.plot('sac254_raw') # doctest: +SKIP

.. plot::
   :context: close-figs
   :include-source: False
   :class: center
   :caption: Result from applying :py:meth:`~saqc.SaQC.flagUniLOF` again on the results for default parameter configuration, this time setting :py:attr:`thresh` parameter to *1.3*.

   qc = qc.flagUniLOF('sac254_raw', thresh=1.3, label='threshold=1.3')
   qc.plot('sac254_raw', xscope=slice('2016-09','2016-11'))

It seems we could sift out some more of the outlier like, flickering values.
Lets lower the threshold even more:

.. doctest:: flagUniLOFExample

   >>> qc = qc.flagUniLOF('sac254_raw', thresh=1.1, label='threshold = 1.1')
   >>> qc.plot('sac254_raw') # doctest: +SKIP

.. plot::
   :context: close-figs
   :include-source: False
   :class: center
   :caption: Even more values get flagged with :py:attr:`thresh=1.1`

   qc = qc.flagUniLOF('sac254_raw', thresh=1.1, label='thresh=1.1')
   qc.plot('sac254_raw', xscope=slice('2016-09','2016-11'))

.. doctest:: flagUniLOFExample

   >>> qc = qc.flagUniLOF('sac254_raw', thresh=1.05, label='threshold = 1.05')
   >>> qc.plot('sac254_raw') # doctest: +SKIP

.. plot::
   :context: close-figs
   :include-source: False
   :class: center
   :caption: Result begins to look overflagged with :py:attr:`thresh=1.05`

   qc = qc.flagUniLOF('sac254_raw', thresh=1.05, label='thresh=1.05')
   qc.plot('sac254_raw', xscope=slice('2016-09','2016-11'))

The lower bound for meaningful values of :py:attr:`thresh` is *1*.
With threshold *1*, the method labels every data point.

.. doctest:: flagUniLOFExample

   >>> qc = qc.flagUniLOF('sac254_raw', thresh=1, label='threshold = 1')
   >>> qc.plot('sac254_raw') # doctest: +SKIP

.. plot::
   :context: close-figs
   :include-source: False
   :class: center
   :caption: Setting :py:attr:`thresh=1` will assign flag to all the values.

   qc = qc.flagUniLOF('sac254_raw', thresh=1, label='thresh=1')
   qc.plot('sac254_raw', xscope=slice('2016-09','2016-11'))

Iterating until `1.1`, seems to give quite a good overall flagging result:

.. plot::
   :context: close-figs
   :include-source: False
   :class: center
   :caption: Overall the outlier detection with :py:attr:`thresh=1.1` seems to work very well. Ideally of course, we would evaluate this result against a validated set of flags while tweaking the parameters.

   qc = saqc.SaQC(data)
   qc = qc.flagUniLOF('sac254_raw', thresh=1.5, label='thresh=1.5')
   qc = qc.flagUniLOF('sac254_raw', thresh=1.3, label='thresh=1.3')
   qc = qc.flagUniLOF('sac254_raw', thresh=1.1, label='thresh=1.1')
   qc.plot('sac254_raw')

The plot shows some over flagging in the closer vicinity of
erratic data jumps.
We will see in the next section, how to fine-tune the algorithm by
shrinking the locality value :py:attr:`n` to make the process more
robust in the surroundings of anomalies.

Before this, lets briefly check on this outlier cluster, at march 2016, that got correctly flagged, as well.

.. plot::
   :context: close-figs
   :include-source: False
   :class: center
   :caption: :py:meth:`~saqc.SaQC.flagUniLOF` will reliably flag groups of outliers, with less than :py:attr:`n/2` periods.

   qc.plot('sac254_raw', xscope=slice('2016-03-15','2016-03-17'))

Tuning Locality Parameter
-------------------------

The parameter :py:attr:`n` controls the number of nearest neighbors
included into the LOF calculation. So :py:attr:`n` effectively
determines the size of the "neighborhood", a data point is compared with, in
order to obtain its "outlierishnes".

Smaller values of :py:attr:`n` can lead to clearer results, because of
feedback effects between normal points and outliers getting mitigated:

.. doctest:: flagUniLOFExample

   >>> qc = saqc.SaQC(data)
   >>> qc = qc.flagUniLOF('sac254_raw', thresh=1.5, n=8, label='thresh=1.5, n= 8')
   >>> qc.plot('sac254_raw', xscope=slice('2016-09','2016-11')) # doctest: +SKIP

.. plot::
   :context: close-figs
   :include-source: False
   :class: center
   :caption: Result with :py:attr:`n=8` and :py:attr:`thresh=20`

   qc = saqc.SaQC(data)
   qc = qc.flagUniLOF('sac254_raw', n=8)
   qc.plot('sac254_raw', xscope=slice('2016-09','2016-11'))

Since :py:attr:`n` determines the size of the surrounding,
a point is compared to, it also determines the maximal size of
detectable outlier clusters. The group we were able to detect by applying :py:meth:`~saqc.SaQC.flagUniLOF`
with :py:attr:`n=20`, is not flagged with :py:attr:`n=8`:

.. plot::
   :context: close-figs
   :include-source: False
   :class: center
   :caption: A cluster with more than :py:attr:`n/2` members, will likely not be detected by the algorithm.

   qc.plot('sac254_raw', xscope=slice('2016-03-15','2016-03-17'))

Also note, that, when changing :py:attr:`n`, you usually have to restart
calibrating a good starting point for the py:attr:`thresh` parameter as well.

Increasingly higher values of :py:attr:`n` will
make :py:meth:`~saqc.SaQC.flagUniLOF` increasingly invariant to local
variance and make it more of a global outlier detection function.
So, an approach towards clearing an entire timeseries from outliers is to start with large :py:attr:`n` to
clear the data from global outliers first, before fine-tuning :py:attr:`thresh` for smaller values of :py:attr:`n` in a second application of the algorithm.

.. doctest:: flagUniLOFExample

   >>> qc = saqc.SaQC(data)
   >>> qc = qc.flagUniLOF('sac254_raw', thresh=1.5, n=100, label='thresh=1.5, n=100')
   >>> qc.plot('sac254_raw')# doctest: +SKIP

.. plot::
   :context: close-figs
   :include-source: False
   :class: center

   qc = saqc.SaQC(data)
   qc = qc.flagUniLOF('sac254_raw', thresh=1.5, n=100, label='thresh=1.5, n=100')
   qc.plot('sac254_raw')



