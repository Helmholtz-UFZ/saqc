
Offset Strings
--------------

All the `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ ars supported by SaQC. The following table lists some of the more relevant options:

.. list-table::
   :header-rows: 1

   * - Alias
     - Description
   * - ``"S"``\ , ``"s"``
     - second
   * - ``"T"``\ , ``"Min"``\ , ``"min"``
     - minute
   * - ``"H"``\ , ``"h"``
     - hour
   * - ``"D"``\ , ``"d"``
     - calendar day
   * - ``"W"``\ , ``"w"``
     - week
   * - ``"M"``\ , ``"m"``
     - month
   * - ``"Y"``\ , ``"y"``
     - year


Multiples are build by preceeding the alias with the desired multiply (e.g ``"5Min"``\ , ``"4W"``\ )

Constants
---------

Flagging Constants
^^^^^^^^^^^^^^^^^^

The following flag constants are available and can be used to mark the quality of a data point:

.. list-table::
   :header-rows: 1

   * - Alias
     - Description
   * - ``GOOD``
     - A value did pass all the test and is therefore considered to be valid
   * - ``BAD``
     - At least on test failed on the values and is therefore considered to be invalid
   * - ``UNFLAGGED``
     - The value has not got a flag yet. This might mean, that all tests passed or that no tests ran


How these aliases will be translated into 'real' flags (output of SaQC) dependes on the :doc:`flagging scheme <FlaggingSchemes>`
and might range from numerical values to string constants.

Numerical Constants
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Alias
     - Description
   * - ``NAN``
     - Not a number

