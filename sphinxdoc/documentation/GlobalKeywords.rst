.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

GlobalKeywords
==============

0. `Example data`_
1. `label`_
    * `label Example Usage`_
2. `dfilter`_
3. `flag`_

Example Data
------------

.. plot::
   :context: close-figs
   :include-source: False

   import matplotlib.pyplot as plt
   import pandas as pd
   import numpy as np
   import saqc
   noise = np.random.normal(0, 1, 200)
   data = pd.Series(noise, index=pd.date_range('2020','2021',periods=200), name='data')
   data.iloc[20] = 16
   data.iloc[100] = -17
   data.iloc[160:180] = -3
   qc = saqc.SaQC(data)

Lets generate some example data and plot it:

.. doctest:: exampleLabel

   >>> import pandas as pd
   >>> import numpy as np
   >>> noise = np.random.normal(0, 1, 200) # some normally distributed noise
   >>> data = pd.Series(noise, index=pd.date_range('2020','2021',periods=200), name='data') # index the noise with some dates
   >>> data.iloc[20] = 16 # add some artificial anomalies:
   >>> data.iloc[100] = -17
   >>> data.iloc[160:180] = -3
   >>> qc = saqc.SaQC(data)
   >>> qc.plot('data') #doctest:+SKIP

.. plot::
   :context: close-figs
   :include-source: False

   qc.plot('data')

Label
-----

The ``label`` keyword can be passed with any function call and serves as label to be plotted by a subsequent
call to :py:meth:`saqc.SaQC.plot`.

It is especially useful for enriching figures with custom context information and for making results from
different function calls distinguishable with respect to their purpose and parameterisation.
Check out the following examples.

Now we apply some flagging functions to mark anomalies, at first, without usage of the ``label`` keyword

.. doctest:: exampleLabel

   >>> qc = qc.flagRange('data', max=15)
   >>> qc = qc.flagRange('data', min=-16)
   >>> qc = qc.flagConstants('data', window='2D', thresh=0)
   >>> qc = qc.flagManual('data', mdata=pd.Series('2020-05', index=pd.DatetimeIndex(['2020-03'])))
   >>> qc.plot('data') # doctest:+SKIP

.. plot::
   :context: close-figs
   :include-source: False

   qc = qc.flagRange('data', max=15)
   qc = qc.flagRange('data', min=-16)
   qc = qc.flagConstants('data', window='2D', thresh=0)
   qc = qc.flagManual('data', mdata=pd.Series('2020-05', index=pd.DatetimeIndex(['2020-03'])))
   qc.plot('data')

In the above plot, one might want to discern the two results from the call to :py:meth:`saqc.SaQC.flagRange` with
respect to the parameters they where called with, also, one might want to give some hints about what is the context of
the flags "manually" determined by the call to :py:meth:`saqc.SaQC.flagManual`. Lets repeat the procedure and
enrich the call with this informations by making use of the label keyword:

Label Example Usage
^^^^^^^^^^^^^^^^^^^

.. doctest:: exampleLabel

   >>> qc = saqc.SaQC(data)
   >>> qc = qc.flagRange('data', max=15, label='values < 15')
   >>> qc = qc.flagRange('data', min=-16, label='values > -16')
   >>> qc = qc.flagConstants('data', window='2D', thresh=0, label='values constant longer than 2 days')
   >>> qc = qc.flagManual('data', mdata=pd.Series('2020-05', index=pd.DatetimeIndex(['2020-03'])), label='values collected while sensor maintenance')
   >>> qc.plot('data') # doctest:+SKIP

.. plot::
   :context: close-figs
   :include-source: False

   qc = saqc.SaQC(data)
   qc = qc.flagRange('data', max=15, label='values < 15')
   qc = qc.flagRange('data', min=-16, label='values > -16')
   qc = qc.flagConstants('data', window='2D', thresh=0, label='values constant longer than 2 days')
   qc = qc.flagManual('data', mdata=pd.Series('2020-05', index=pd.DatetimeIndex(['2020-03'])), label='values collected while sensor maintenance')
   qc.plot('data')


dfilter
-------

The ``dfilter`` keyword controls the threshold up to which a flag triggers masking of its associated value, when passed
on, to any flagging function. Any value ``v`` with a flag ``f(v)`` will be masked, if ``f(v) > dfilter``. A masked value
is not visible to a flagging function, so it will neither be part of any calculations performed, nor will it be
flagged by this function. Lets visualize this with the :py:plot:`saqc.SaqC.plot` method. (We are reusing data and code
from `Example Data`_ section). First, we set some flags to the data:

.. doctest:: exampleLabel

   >>> qc = saqc.SaQC(data)
   >>> qc = qc.flagRange('data', max=15, label='flaglevel=200', flag=200)
   >>> qc = qc.flagRange('data', min=-16, label='flaglevel=100', flag=100)
   >>> qc = qc.flagManual('data', mdata=pd.Series('2020-05', index=pd.DatetimeIndex(['2020-03'])), label='flaglevel=0', flag=0)
   >>> qc.plot('data') # doctest:+SKIP


.. plot::
   :context: close-figs
   :include-source: False

   qc = saqc.SaQC(data)
   qc = qc.flagRange('data', max=15, label='flaglevel=200', flag=200)
   qc = qc.flagRange('data', min=-16, label='flaglevel=100', flag=100)
   qc = qc.flagManual('data', mdata=pd.Series('2020-05', index=pd.DatetimeIndex(['2020-03'])), label='flaglevel=0', flag=0)
   qc.plot('data')

With the ``dfilter`` Keyword, we can now control, which of the flags are passed on to the plot function.
For example, if we set ``dfilter=50``, the flags set by the :py:meth:`saqc.SaQC.flagRange` method wont get passed on
and thus, the flagged values wont be visible in the plot:

.. doctest:: exampleLabel

   >>> qc.plot('data', dfilter=50) # doctest:+SKIP

.. plot::
   :context: close-figs
   :include-source: False

   qc.plot('data', dfilter=50)
