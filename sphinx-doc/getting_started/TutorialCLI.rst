Command Line Application
========================

.. note:

    The content of this page is outdate and the provided code sample will very likely
    not work as expected. A rewrite of this document is pending...


Contents
--------


* :ref:`1 Get toy data and configuration <getting_started/GettingStarted:get toy data and configuration>`
* :ref:`2 Run SaQC <getting_started/GettingStarted:run saqc>`
* :ref:`3 Configure SaQC <getting_started/GettingStarted:configure saqc>`

  * :ref:`Change test parameters <getting_started/GettingStarted:change test parameters>`

* :ref:`4 Explore the functionality <getting_started/GettingStarted:explore the functionality>`

  * :ref:`Process multiple variables <getting_started/GettingStarted:process multiple variables>`
  * :ref:`Data harmonization and custom functions <getting_started/GettingStarted:data harmonization and custom functions>`


The following passage guides you through the essentials of the usage of SaQC via
a toy dataset and a toy configuration.

1. Get toy data and configuration
---------------------------------

If you take a look into the folder ``saqc/ressources/data`` you will find a toy
dataset ``data.csv`` which contains the following:

.. code-block::

   Date,Battery,SM1,SM2
   2016-04-01 00:05:48,3573,32.685,29.3157
   2016-04-01 00:20:42,3572,32.7428,29.3157
   2016-04-01 00:35:37,3572,32.6186,29.3679
   2016-04-01 00:50:32,3572,32.736999999999995,29.3679
   ...


These are two timeseries of soil moisture (SM1+2) and the battery voltage of the
measuring device over time. Generally, this is the way that your data should
look like to run saqc. Note, however, that you do not necessarily need a series
of dates to reference to and that you are free to use more columns of any name
that you like.

Now create your our own configuration file ``saqc/ressources/data/myconfig.csv``
and paste the following lines into it:

.. code-block::

   varname;test;plot
   SM2;flagRange(min=10, max=60);False
   SM2;flagMad(window="30d", z=3.5);True


These lines illustrate how different quality control tests can be specified for
different variables,

In this case, we define a range-test that flags all values outside the range
[10,60] and a test to detect spikes using the MAD-method. You can find an
overview of all available quality control tests in the
:doc:`documentation <FunctionIndex>`. Note that the tests are
*executed in the order that you define in the configuration file*. The quality
flags that are set during one test are always passed on to the subsequent one.

2. Run SaQC
-----------

Remember to have your virtual environment activated:

On Unix/Mac-systems
"""""""""""""""""""

.. code-block:: sh

   source env_saqc/bin/activate

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

On Unix/Mac-systems
"""""""""""""""""""

.. code-block:: sh

   python3 -m saqc -c ressources/data/myconfig.csv -d ressources/data/data.csv

On Windows
""""""""""

.. code-block:: sh

   py -3 -m saqc -c ressources/data/myconfig.csv -d ressources/data/data.csv

If you installed saqc via PYPi, you can omit ``sh python -m``.

The command will output this plot:


.. image:: ../ressources/images/example_plot_1.png
   :target: ../ressources/images/example_plot_1.png
   :alt: Toy Plot


So, what do we see here?


* The plot shows the data as well as the quality flags that were set by the
  tests for the variable ``SM2``\ , as defined in the config-file
* Following our definition in the config-file, first the ``flagRange``\ -test that flags
  all values outside the range [10,60] was executed and after that,
  the ``flagMad``\ -test to identify spikes in the data
* In the config, we set the plotting option to ``True`` for ``flagMad``\ ,
  only. Thus, the plot aggregates all preceeding tests (here: ``range``\ ) to black
  points and highlights the flags of the selected test as red points.

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

.. code-block::

   varname;test;plot
   SM2;flagRange(min=-20, max=60);False
   SM2;flagMad(window="30d", z=3.5);True

Rerunning SaQC as above produces the following plot:


.. image:: ../ressources/images/example_plot_2.png
   :target: ../ressources/images/example_plot_2.png
   :alt: Changing the config


You can see that the changes that we made to the parameters of the range test
take effect so that only the values > 60 are flagged by it (black points). This,
in turn, leaves more erroneous data that is then identified by the proceeding
spike-test (red points).

4. Explore the functionality
----------------------------

Process multiple variables
""""""""""""""""""""""""""

You can also define multiple tests for multiple variables in your data. These
are then executed sequentially and can be plotted seperately. E.g. you could do
something like this:

.. code-block::

   varname;test;plot
   SM1;flagRange(min=10, max=60);False
   SM2;flagRange(min=10, max=60);False
   SM1;flagMad(window="15d", z=3.5);True
   SM2;flagMad(window="30d", z=3.5);True


which gives you separate plots for each line where the plotting option is set to
``True`` as well as one summary "data plot" that depicts the joint flags from all
tests:

.. list-table::
   :header-rows: 1

   * - SM1
     - SM2
   * - here
     - there


.. list-table::
   :header-rows: 1

   * - SM1
     - SM2
   * - .. image:: ../ressources/images/example_plot_31.png
          :target: ../ressources/images/example_plot_31.png
          :alt: 
       
     - .. image:: ../ressources/images/example_plot_32.png
          :target: ../ressources/images/example_plot_32.png
          :alt: 
       
   * - .. image:: ../ressources/images/example_plot_31.png
          :target: ../ressources/images/example_plot_31.png
          :alt: 
       
     -



.. image:: ../ressources/images/example_plot_33.png
   :target: ../ressources/images/example_plot_33.png
   :alt: 


Data harmonization and custom functions
"""""""""""""""""""""""""""""""""""""""

SaQC includes functionality to harmonize the timestamps of one or more data
series. Also, you can write your own tests using a python-based
:doc:`extension language <GenericFunctions>`. This would look like this:

.. code-block::

   varname;test;plot
   SM2;shiftToFreq(freq="15Min");False
   SM2;generic(func=(SM2 < 30));True


The above executes an internal framework that harmonizes the timestamps of SM2
to a 15min-grid (see data below). Further information about this routine can be
found in the :ref:`Flagging Functions Overview <flaggingFunctions>`.

.. code-block::

   Date,SM1,SM1_flags,SM2,SM2_flags
   2016-04-01 00:00:00,,,29.3157,OK
   2016-04-01 00:05:48,32.685,OK,,
   2016-04-01 00:15:00,,,29.3157,OK
   2016-04-01 00:20:42,32.7428,OK,,
   ...


Also, all values where SM2 is below 30 are flagged via the custom function (see
plot below). You can learn more about the syntax of these custom functions
:doc:`here <GenericFunctions>`.


.. image:: ../ressources/images/example_plot_4.png
   :target: ../ressources/images/example_plot_4.png
   :alt: Example custom function

