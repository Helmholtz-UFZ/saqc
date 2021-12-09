
.. testsetup:: default

   datapath = './ressources/data/hydro_data.csv'
   maintpath = './ressources/data/hydro_maint.csv'

.. plot::
   :context:
   :include-source: False

   import matplotlib
   import saqc
   import pandas as pd
   datapath = '../ressources/data/hydro_data.csv'
   maintpath = '../ressources/data/hydro_maint.csv'
   data = pd.read_csv(datapath, index_col=0)
   maint = pd.read_csv(maintpath, index_col=0)
   maint.index = pd.DatetimeIndex(maint.index)
   data.index = pd.DatetimeIndex(data.index)
   qc = saqc.SaQC([data, maint])


Multivariate Flagging
=====================

The tutorial aims to introduce the usage of SaQC in the context of some more complex flagging and processing techniques. 
Mainly we will see how to apply Drift Corrections onto the data and how to perform multivariate flagging.


#. `Data Preparation`_

#. `Drift Correction`_


Data Preparation
----------------

First import the data (from the repository), and generate an saqc instance from it. You will need to download the `sensor
data <https://git.ufz.de/rdm-software/saqc/-/blob/develop/sphinxdoc/ressources/data/hydro_config.csv>`_ and the
`maintenance data <https://git.ufz.de/rdm-software/saqc/-/blob/develop/sphinxdoc/ressources/data/hydro_maint.csv>`_
from the `repository <https://git.ufz.de/rdm-software/saqc.git>`_ and make variables `datapath` and `maintpath` be
paths pointing at those downloaded files. Note, that the :py:class:`~saqc.SaQC` digests the loaded data in a list.
This is done, to circumvent having to concatenate both datasets in a pandas Dataframe instance, which would introduce
`NaN` values to both the datasets, wherever their timestamps missmatch. `SaQC` can handle those unaligned data
internally without introducing artificially fill values to them.

.. testcode:: default

   data = pd.read_csv(datapath, index_col=0)
   maint = pd.read_csv(maintpath, index_col=0)
   maint.index = pd.DatetimeIndex(maint.index)
   data.index = pd.DatetimeIndex(data.index)
   qc = saqc.SaQC([data, maint])  # dataframes "data" and "maint" are integrated internally

We can check out the fields, the newly generated :py:class:`~saqc.SaQC` object contains as follows:

.. doctest:: default

   >>> qc.data.columns
   Index(['sac254_raw', 'level_raw', 'water_temp_raw', 'maint'], dtype='object', name='columns')

The variables represent meassurements of *water level*, the *specific absorption coefficient* at 254 nm Wavelength,
the *water temperature* and there is also a variable, *maint*, that refers to time periods, where the *sac254* sensor
was maintained. Lets have a look at those:

.. doctest:: default

   >>> qc.data_raw['maint'] # doctest:+SKIP
   Timestamp
   2016-01-10 11:15:00    2016-01-10 12:15:00
   2016-01-12 14:40:00    2016-01-12 15:30:00
   2016-02-10 13:40:00    2016-02-10 14:40:00
   2016-02-24 16:40:00    2016-02-24 17:30:00
   ....                                  ....
   2017-10-17 08:55:00    2017-10-17 10:20:00
   2017-11-14 15:30:00    2017-11-14 16:20:00
   2017-11-27 09:10:00    2017-11-27 10:10:00
   2017-12-12 14:10:00    2017-12-12 14:50:00
   Name: maint, dtype: object

Measurements collected while maintenance are not trustworthy, so any measurement taken, in any of the listed
intervals should be flagged right away. This can be achieved, with the :py:meth:`~saqc.SaQC.flagManual` method. Also,
we will flag out-of-range values in the data with the :py:meth:`~saqc.SaQC.flagRange` method:

.. doctest:: default

   >>> qc = qc.flagManual('sac254_raw', mdata='maint', method='closed', label='Maintenance')
   >>> qc = qc.flagRange('level_raw', min=0)
   >>> qc = qc.flagRange('water_temp_raw', min=-1, max=40)
   >>> qc = qc.flagRange('sac254_raw', min=0, max=60)

.. plot::
   :context:
   :include-source: False

   qc = qc.flagManual('sac254_raw', mdata='maint', method='closed', label='Maintenance')
   qc = qc.flagRange('level_raw', min=0)
   qc = qc.flagRange('water_temp_raw', min=-1, max=40)
   qc = qc.flagRange('sac254_raw', min=0, max=60)

Lets check out the resulting flags for the *sac254* variable with the :py:meth:`~saqc.SaQC.plot` method:

.. plot::
   :context:
   :include-source: True
   :format: doctest

   >>> qc.plot('sac254_raw') #doctest:+SKIP

Now we should figure out, what sampling rate the data is intended to have, by accessing the *_raw* variables
constituting the sensor data. Since :py:attr:`saqc.SaQC.data` yields a common
`pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ object, we can index it with
the desired variables as column names and have a look at the console output to get a first impression.

.. doctest:: default

   >>> qc.data[['sac254_raw', 'level_raw', 'water_temp_raw']] # doctest:+NORMALIZE_WHITESPACE
   columns              sac254_raw  level_raw  water_temp_raw
   Timestamp
   2016-01-01 00:02:00     18.4500    103.290            4.84
   2016-01-01 00:17:00     18.6437    103.285            4.82
   2016-01-01 00:32:00     18.9887    103.253            4.81
   2016-01-01 00:47:00     18.8388    103.210            4.80
   2016-01-01 01:02:00     18.7438    103.167            4.78
                            ...        ...             ...
   2017-12-31 22:47:00     43.2275    186.060            5.49
   2017-12-31 23:02:00     43.6937    186.115            5.49
   2017-12-31 23:17:00     43.6012    186.137            5.50
   2017-12-31 23:32:00     43.2237    186.128            5.51
   2017-12-31 23:47:00     43.7438    186.130            5.53
   <BLANKLINE>
   [70199 rows x 3 columns]

The data seems to have a fairly regular sampling rate of *15* minutes at first glance.
But checking out values around *2017-10-29*, we notice, that the sampling rate seems not to be totally stable:

.. doctest:: default

   >>> qc.data[['sac254_raw', 'level_raw', 'water_temp_raw']]['2017-10-29 07:00:00':'2017-10-29 09:00:00'] # doctest:+NORMALIZE_WHITESPACE
   columns              sac254_raw  level_raw  water_temp_raw
   Timestamp
   2017-10-29 07:02:00     40.3050    112.570           10.91
   2017-10-29 07:17:00     39.6287    112.497           10.90
   2017-10-29 07:32:00     39.5800    112.460           10.88
   2017-10-29 07:32:01     39.9750    111.837           10.70
   2017-10-29 07:47:00     39.1350    112.330           10.84
   2017-10-29 07:47:01     40.6937    111.615           10.68
   2017-10-29 08:02:00     40.4938    112.040           10.77
   2017-10-29 08:02:01     39.3337    111.552           10.68
   2017-10-29 08:17:00     41.5238    111.835           10.72
   2017-10-29 08:17:01     38.6963    111.750           10.69
   2017-10-29 08:32:01     39.4337    112.027           10.66
   2017-10-29 08:47:01     40.4987    112.450           10.64

Those instabilities do bias most statistical evaluations and it is common practice to apply some
:doc:`resampling functions <../funcSummaries/resampling>` onto the data, to obtain a regularly spaced timestamp.
(See also the :ref:`harmonization tutorial <cook_books/DataRegularisation:data regularisation>` for more informations
on that topic.)

We will apply :py:meth:`linear harmonisation <saqc.SaQC.linear>` to all the sensor data variables,
to interpolate pillar points of multiples of *15* minutes linearly.

.. doctest:: default

   >>> qc = qc.linear(['sac254_raw', 'level_raw', 'water_temp_raw'], freq='15min')

.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.linear(['sac254_raw', 'level_raw', 'water_temp_raw'], freq='15min')


The resulting timeseries has regular timestamp and includes only values that evaluate to `NaN` or did pass the range
check and the maintenance data flagging:


.. doctest:: default

   >>> qc.data['sac254_raw'] #doctest:+NORMALIZE_WHITESPACE
   Timestamp
   2016-01-01 00:00:00          NaN
   2016-01-01 00:15:00    18.617873
   2016-01-01 00:30:00    18.942700
   2016-01-01 00:45:00    18.858787
   2016-01-01 01:00:00    18.756467
                            ...
   2017-12-31 23:00:00    43.631540
   2017-12-31 23:15:00    43.613533
   2017-12-31 23:30:00    43.274033
   2017-12-31 23:45:00    43.674453
   2018-01-01 00:00:00          NaN
   Name: sac254_raw, Length: 70194, dtype: float64

.. plot::
   :context:
   :include-source: True
   :format: doctest

   >>> qc.plot('sac254_raw') # doctest:+SKIP


Drift Correction
----------------

The variables *SAK254* and *Turbidity* show drifting behavior originating from dirt, that accumulates on the light
sensitive sensor surfaces over time. The effect, the dirt accumulation has on the measurement values, is assumed to be
properly described by an exponential model. The Sensors are cleaned periodocally, resulting in a periodical reset of
the drifting effect. The Dates and Times of the maintenance events are input to the
:py:meth:`~saqc.SaQC.correctDrift>`, that will correct the data in between any two such maintenance intervals.

.. doctest:: default

   >>> qc = qc.correctDrift('sac254_raw', target='sac254_corrected',maintenance_field='maint', model=expDriftModel)

.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.correctDrift('sac254_raw', target='sac254_corrected',maintenance_field='maint', model='exponential')

Check out results

.. plot::
   :context:
   :include-source: True
   :format: doctest

   >>> plt.plot(qc.data_raw['sac254_raw'])
   >>> plt.plot(qc.data_raw['sac254_corrected'])

Apply Multivariate Flagging
---------------------------

We are basically following the *oddWater* procedure, as suggested in *Talagala, P.D. et al (2019): A Feature-Based
Procedure for Detecting Technical Outliers in Water-Quality Data From In Situ Sensors. Water Ressources Research,
55(11), 8547-8568.*

First we define a transformation we want the variables to be normalized with.
We just import *scipys* `zscore` function and wrap it, so that it will
be able to digest *nan* values without returning *nan*

.. testcode:: default

   from scipy.stats import zscore
   zscore_func = lambda x: zscore(x, nan_policy='omit')

.. plot::
   :context: close-figs
   :include-source: False

   from scipy.stats import zscore
   zscore_func = lambda x: zscore(x, nan_policy='omit')

Now we can pass the function to the :py:meth:`saqc.SaQC.transform` method.

.. testcode:: default

   qc = qc.transform(['sac254_raw', 'level_raw', 'water_temp_raw'], target=['sac_z', 'level_z', 'water_z'], func=zscore_func, freq='30D')

.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.transform(['sac254_raw', 'level_raw', 'water_temp_raw'], target=['sac_z', 'level_z', 'water_z'], func=zscore_func, freq='30D')

The idea of the *oddWater* algorithm, is, to assign any timestamp a score, derived from the distance of the *k* nearest
neighbors of the datapoint related to that score. We can do this, via the :py:meth:`~saqc.SaQC.assignKNNScores` method.

.. testsetup:: default

   qc = qc.assignKNNScore(field=['sac254_z', 'level_z', 'water_temp_z'], target='kNNscores', freq='30D', n=5)

.. plot::
   :context: close-figs
   :include-source: True
   :format: doctest

   >>> qc = qc.assignKNNScore(['sac254_z', 'level_z', 'water_temp_z'], target='kNNscores', freq='30D', n=5)
   >>> qc.plot('kNNscores') # doctest:+SKIP

Those scores roughly correlate with the isolation of the scored points in the phase space. For example, have a look at
the phase space of *sac* and *level*

.. plot::
   :context: close-figs
   :include-source: True
   :format: doctest

   >>> qc.plot('sac_z', phaseplot='level_z') # doctest:+SKIP

* Variables *SAK254*\ , *Turbidity*\ , *Pegel*\ , *NO3N*\ , *WaterTemp* and *pH* get transformed to comparable scales
* We are obtaining nearest neighbor scores and assigign those to a new variable, via :py:func:`assignKNNScores <Functions.saqc.assignKNNScores>`.
* We are applying the *STRAY* Algorithm to find the cut_off points for the scores, above which values qualify as outliers. (:py:func:`flagByStray <Functions.saqc.flagByStray>`)
* We project the calculated flags onto the input variables via :py:func:`assignKNNScore <Functions.saqc.assignKNNScore>`.

Postprocessing
--------------


* (Flags reduction onto subspaces)
* Back projection of calculated flags from resampled Data onto original data via :py:func: ``mapToOriginal <Functions.saqc.mapToOriginal>``
