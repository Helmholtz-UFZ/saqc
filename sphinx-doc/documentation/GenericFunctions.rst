
Generic Functions
=================

Generic Flagging Functions
--------------------------

Generic flagging functions provide for cross-variable quality
constraints and to implement simple quality checks directly within the configuration.

Why?
^^^^

In most real world datasets many errors
can be explained by the dataset itself. Think of a an active, fan-cooled
measurement device: no matter how precise the instrument may work, problems
are to be expected when the fan stops working or the power supply 
drops below a certain threshold. While these dependencies are easy to 
:ref:`formalize <documentation/GenericFunctions:a real world example>` on a per dataset basis, it is quite
challenging to translate them into generic source code.

Specification
^^^^^^^^^^^^^

Generic flagging functions are used in the same manner as their
:doc:`non-generic counterparts <FunctionIndex>`. The basic 
signature looks like that:

.. code-block:: sh

   flagGeneric(func=<expression>, flag=<flagging_constant>)

where ``<expression>`` is composed of the `supported constructs`_
and ``<flag_constant>`` is one of the predefined
:ref:`flagging constants <getting_started/ParameterDescriptions:flagging constants>` (default: ``BAD``\ ).
Generic flagging functions are expected to return a boolean value, i.e. ``True`` or ``False``. All other expressions will
fail during the runtime of ``SaQC``.

Examples
^^^^^^^^

Simple comparisons
~~~~~~~~~~~~~~~~~~

Task
""""

Flag all values of ``x`` where ``y`` falls below 0.

Configuration file
""""""""""""""""""

.. code-block::

   varname ; test                    
   #-------;------------------------
   x       ; flagGeneric(func=y < 0)

Calculations
~~~~~~~~~~~~

Task
""""

Flag all values of ``x`` that exceed 3 standard deviations of ``y``.

Configuration file
""""""""""""""""""

.. code-block::

   varname ; test
   #-------;---------------------------------
   x       ; flagGeneric(func=x > std(y) * 3)

Special functions
~~~~~~~~~~~~~~~~~

Task
""""

Flag all values of ``x`` where: ``y`` is flagged and ``z`` has missing values.

Configuration file
""""""""""""""""""

.. code-block::

   varname ; test
   #-------;----------------------------------------------
   x       ; flagGeneric(func=isflagged(y) & ismissing(z))

A real world example
~~~~~~~~~~~~~~~~~~~~

Let's consider the following dataset:

.. list-table::
   :header-rows: 1

   * - date
     - meas
     - fan
     - volt
   * - 2018-06-01 12:00
     - 3.56
     - 1
     - 12.1
   * - 2018-06-01 12:10
     - 4.7
     - 0
     - 12.0
   * - 2018-06-01 12:20
     - 0.1
     - 1
     - 11.5
   * - 2018-06-01 12:30
     - 3.62
     - 1
     - 12.1
   * - ...
     - 
     - 
     - 


Task
""""

Flag ``meas`` where ``fan`` equals 0 and ``volt``
is lower than ``12.0``.

Configuration file
""""""""""""""""""

There are various options. We can directly implement the condition as follows:

.. code-block::

   varname ; test
   #-------;-----------------------------------------------
   meas    ; flagGeneric(func=(fan == 0) \|  (volt < 12.0))

But we could also quality check our independent variables first
and than leverage this information later on:

.. code-block::

   varname ; test
   #-------;----------------------------------------------------
   '.*'    ; flagMissing()
   fan     ; flagGeneric(func=fan == 0)
   volt    ; flagGeneric(func=volt < 12.0)
   meas    ; flagGeneric(func=isflagged(fan) \| isflagged(volt))

Generic Processing
------------------

Generic processing functions provide a way to evaluate mathmetical operations 
and functions on the variables of a given dataset.

Why
^^^

In many real-world use cases, quality control is embedded into a larger data 
processing pipeline and it is not unusual to even have certain processing 
requirements as a part of the quality control itself. Generic processing 
functions make it easy to enrich a dataset through the evaluation of a
given expression.

Specification
^^^^^^^^^^^^^

The basic signature looks like that:

.. code-block:: sh

   procGeneric(func=<expression>)

where ``<expression>`` is composed of the `supported constructs`_.

Variable References
-------------------

All variables of the processed dataset are available within generic functions,
so arbitrary cross references are possible. The variable of interest 
is furthermore available with the special reference ``this``\ , so the second 
:ref:`example <documentation/GenericFunctions:calculations>` could be rewritten as:

.. code-block::

   varname ; test
   #-------;------------------------------------
   x       ; flagGeneric(func=this > std(y) * 3)

When referencing other variables, their flags will be respected during evaluation
of the generic expression. So, in the example above only values of ``x`` and ``y``\ , that
are not already flagged with ``BAD`` will be used the avaluation of ``x > std(y)*3``. 

Supported constructs
--------------------

Operators
^^^^^^^^^

Comparison
~~~~~~~~~~

The following comparison operators are available:

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
   * - ``==``
     - ``True`` if the values of the operands are equal
   * - ``!=``
     - ``True`` if the values of the operands are not equal
   * - ``>``
     - ``True`` if the values of the left operand are greater than the values of the right operand
   * - ``<``
     - ``True`` if the values of the left operand are smaller than the values of the right operand
   * - ``>=``
     - ``True`` if the values of the left operand are greater or equal than the values of the right operand
   * - ``<=``
     - ``True`` if the values of the left operand are smaller or equal than the values of the right operand


Arithmetics
~~~~~~~~~~~

The following arithmetic operators are supported:

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
   * - ``+``
     - addition
   * - ``-``
     - subtraction
   * - ``*``
     - multiplication
   * - ``/``
     - division
   * - ``**``
     - exponentiation
   * - ``%``
     - modulus


Bitwise
~~~~~~~

The bitwise operators also act as logical operators in comparison chains

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
   * - ``&``
     - binary and
   * - ``|``
     - binary or
   * - ``^``
     - binary xor
   * - ``~``
     - binary complement


Functions
^^^^^^^^^

All functions expect a :ref:`variable reference <documentation/GenericFunctions:variable references>`
as the only non-keyword argument (see :ref:`here <documentation/GenericFunctions:special functions>`\ )

Mathematical Functions
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - ``abs``
     - absolute values of a variable
   * - ``max``
     - maximum value of a variable
   * - ``min``
     - minimum value of a variable
   * - ``mean``
     - mean value of a variable
   * - ``sum``
     - sum of a variable
   * - ``std``
     - standard deviation of a variable
   * - ``len``
     - the number of values for variable


Special Functions
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - ``ismissing``
     - check for missing values
   * - ``isflagged``
     - check for flags


Constants
^^^^^^^^^

Generic functions support the same constants as normal functions, a detailed 
list is available :ref:`here <getting_started/ParameterDescriptions:constants>`.
