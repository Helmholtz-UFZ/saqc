.. testsetup:: exampleCLI

   datapath = './ressources/data/data.csv'
   configpath = lambda x: f'./ressources/data/myconfig{x}.csv'
   temppath = lambda x: f'./ressources/temp/{x}'
   data = pd.read_csv(datapath, index_col=0)
   data.index = pd.DatetimeIndex(data.index)

.. plot::
   :include-source: False
   :context: close-figs

   import pandas as pd
   import saqc
   import matplotlib
   datapath = '../ressources/data/data.csv'
   configpath = lambda x: f'../ressources/data/myconfig{x}.csv'
   temppath = lambda x: f'../ressources/temp/{x}'
   data = pd.read_csv(datapath, index_col=0)
   data.index = pd.DatetimeIndex(data.index)

Command Line Application
========================


Contents
--------


* `1. Get toy data and configuration`_
* `2. Run SaQC`_
* `3. Configure SaQC`_

  * `Change test parameters`_

* `4. Explore the functionality`_

  * `Process multiple variables`_
  * `Data harmonization and custom functions`_


The following passage guides you through the essentials of the usage of SaQC via
a toy dataset and a toy configuration.

1. Get toy data and configuration
---------------------------------

If you take a look into the folder ``saqc/ressources/data`` you will find a toy
dataset ``data.csv`` which contains the following data:

.. literalinclude:: ../ressources/data/data.csv
   :lines: 1-6


These are the first entries of two timeseries of soil moisture (SM1+2) and the battery voltage of the
measuring device over time. Generally, this is the way that your data should
look like to run saqc. Note, however, that you do not necessarily need a series
of dates to reference to and that you are free to use more columns of any name
that you like.

Now have a look at a basic sonfiguration file, as
`this one <https://git.ufz.de/rdm-software/saqc/-/blob/develop/sphinxdoc/ressources/data/myconfig.csv>`_.
It contains the following lines:

.. literalinclude:: ../ressources/data/myconfig.csv

These lines illustrate how different quality control tests can be specified for
different variables.

In this case, we trigger a :py:meth:`range <saqc.SaQC.flagRange>` test, that flags all values exceeding
the range of the bounds of the interval *[10,60]*. Subsequently, a test to detect spikes, is applied,
using the MAD-method. (:py:meth:`~saqc.SaQC.flagMAD`).
You can find an overview of all available quality control tests
:ref:`here <moduleAPI/saqcFuncTOC:SaQC Flagging Functions>`. Note that the tests are
*executed in the order that you define in the configuration file*. The quality
flags that are set during one test are always passed on to the subsequent one.

.. testcode:: exampleCLI
   :hide:

   qc = saqc.fromConfig(configpath(''), data)

2. Run SaQC
-----------

On Unix/Mac-systems
"""""""""""""""""""

Remember to have your virtual environment activated:

.. code-block:: sh

   source env_saqc/bin/activate

From here, you can run saqc and tell it to run the tests from the toy
config-file on the toy dataset via the ``-c`` and ``-d`` options:

.. code-block:: sh

   python3 -m saqc -c ressources/data/myconfig.csv -d ressources/data/data.csv

On Windows
""""""""""

.. code-block:: sh

   cd env_saqc/Scripts
   ./activate

Via your console, move into the folder you downloaded saqc into:

.. code-block:: sh

   cd saqc

From here, you can run saqc and tell it to run the tests from the toy
config-file on the toy dataset via the ``-c`` and ``-d`` options:


.. code-block:: sh

   py -3 -m saqc -c ressources/data/myconfig.csv -d ressources/data/data.csv

If you installed saqc via PYPi, you can omit ``sh python -m``.

The command will output this plot:


.. plot::
   :context: close-figs
   :include-source: False
   :width: 80 %
   :align: center

   qc = saqc.fromConfig(configpath(''), data)

So, what do we see here?


* The plot shows the data as well as the quality flags that were set by the
  tests for the variable ``SM2``\ , as defined in the config-file
* Following our definition in the config-file, first the :py:meth:`~saqc.SaQC.flagRange` -test that flags
  all values outside the range [10,60] was executed and after that,
  the :py:meth:`~saqc.SaQC.flagMAD` -test to identify spikes in the data
* Finally we triggered the generation of a plot, by adding the :py:meth:`~saqc.SaQC.plot` function in the last line.

Save outputs to file
^^^^^^^^^^^^^^^^^^^^

If you want the final results to be saved to a csv-file, you can do so by the
use of the ``-o`` option:

.. code-block:: sh

   saqc -c ressources/data/config.csv -d ressources/data/data.csv -o ressources/data/out.csv

Which saves a dataframe that contains both the original data and the quality
flags that were assigned by SaQC for each of the variables:

.. code-block::

   Date,SM1,SM1_flags,SM2,SM2_flags
   2016-04-01 00:05:48,32.685,OK,29.3157,OK
   2016-04-01 00:20:42,32.7428,OK,29.3157,OK
   2016-04-01 00:35:37,32.6186,OK,29.3679,OK
   2016-04-01 00:50:32,32.736999999999995,OK,29.3679,OK
   ...



3. Configure SaQC
-----------------

Change test parameters
""""""""""""""""""""""

Now you can start to change the settings in the config-file and investigate the
effect that has on how many datapoints are flagged as "BAD". When using your
own data, this is your way to configure the tests according to your needs. For
example, you could modify your ``myconfig.csv`` and change the parameters of the
range-test:

.. literalinclude:: ../ressources/data/myconfig2.csv

Rerunning SaQC as above produces the following plot:

.. testcode:: exampleCLI
   :hide:

   qc = saqc.fromConfig(configpath('2'), data)


.. plot::
   :context: close-figs
   :include-source: False
   :width: 80 %
   :align: center

   qc = saqc.fromConfig(configpath('2'), data)


You can see that the changes that we made to the parameters of the range test
take effect so that only the values > 60 are flagged by it (black points). This,
in turn, leaves more erroneous data that is then identified by the proceeding
spike-test (red points).

4. Explore the functionality
----------------------------

Process multiple variables
""""""""""""""""""""""""""

You can also define multiple tests for multiple variables in your data. These
are then executed sequentially and can be plotted seperately. To not interrupt processing, the plots
get stored to files.

.. literalinclude:: ../ressources/data/myconfig4.csv


.. plot::
   :context: close-figs
   :include-source: False

   qc = saqc.fromConfig(configpath('4'), data)

which gives you separate plots for each line where the plotting option is set to
``True`` as well as one summary "data plot" that depicts the joint flags from all
tests:


.. list-table::
   :header-rows: 1

   * - SM1
     - SM2
   * - .. image:: ../ressources/images/SM1processingResults.png
          :target: ../ressources/images/SM1processingResults.png
          :alt: 
       
     - .. image:: ../ressources/images/SM2processingResults.png
          :target: ../ressources/images/SM2processingResults.png
          :alt: 


Data harmonization and custom functions
"""""""""""""""""""""""""""""""""""""""

SaQC includes functionality to harmonize the timestamps of one or more data
series. Also, you can write your own tests using a python-based
:ref:`extension language <documentation/GenericFunctions:Generic Functions>`. This would look like this:

.. literalinclude:: ../ressources/data/myconfig3.csv

.. testcode:: exampleCLI
   :hide:

   qc = saqc.fromConfig(configpath('3'), data)

.. plot::
   :context: close-figs
   :include-source: False
   :nofigs:

   import os
   qc = saqc.fromConfig(configpath('3'), data)
   qc.data.to_csv(temppath('TutorialCLIHarmData.csv'))


The above executes an internal framework that aligns the timestamps of SM2
to a 15min-grid (:py:meth:`saqc.SaQC.shift`). Further information on harmonization can be
found in the :doc:`Resampling cookbook <../cook_books/DataRegularisation>`.


.. literalinclude:: ../ressources/temp/TutorialCLIHarmData.csv
   :lines: 1-10


Also, all values where SM2 is below 30 are flagged via the custom function (see
plot below) and the plot is labeled with the string passed to the `label` keyword.
You can learn more about the syntax of these custom functions
:ref:`here <documentation/GenericFunctions:Generic Functions>`.


.. plot::
   :context: close-figs
   :include-source: False
   :width: 80 %
   :align: center

   qc.plot('SM2')
