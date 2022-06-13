.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Generic Functions
=================

Generic Flagging Functions
--------------------------

Generic flagging functions provide for custom cross-variable quality constraints, directly
implemented using the :doc:`Python API <../gettingstarted/TutorialAPI>` or the
:doc:`Configuration System <../documentation/ConfigurationFiles>`.

Why?
^^^^

In most real world datasets many errors can be explained by the dataset itself. Think of a an
active, fan-cooled measurement device: no matter how precise the instrument may work, problems
are to be expected when the fan stops working or the power supply drops below a certain threshold.
While these dependencies are easy to :ref:`formalize <documentation/GenericFunctions:a real world example>`
on a per dataset basis, it is quite challenging to translate them into generic source code. That
is why we instrumented SaQC to cope with such situations.


Generic Flagging - Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generic flagging functions are used in the same manner as their
non-generic counterparts. The basic signature looks like that:

.. code-block::

   flagGeneric(field, func=<expression>, flag=<flag_constant>)

where ``<expression>`` is either a callable (Python API) or an expression
composed of the `supported constructs`_
and ``<flag_constant>`` is either one of the predefined
:ref:`flagging constants <modules/saqcConstants:Flagging Constants>`
(default: ``BAD``\ ) or a valid value of the chosen flagging scheme. Generic flagging functions
are expected to return a collection of boolean values, i.e. one ``True`` or ``False`` for every
value in ``field``. All other expressions will fail during runtime of SaQC.


Examples
^^^^^^^^

The following sections show some contrived but realistic examples, highlighting the
potential of :py:meth:`flagGeneric <saqq.SaQC.flagGeneric>`. Let's first generate a
dummy dataset, to lead us through the following code snippets:

.. testsetup:: python

   from saqc import fromConfig
   from tests.common import writeIO

.. testcode:: python
              
   from saqc import SaQC

   x = np.array([12, 87, 45, 31, 18, 99])
   y = np.array([2, 12, 33, 133, 8, 33])
   z = np.array([34, 23, 89, 56, 5, 1])

   dates = pd.date_range(start="2020-01-01", periods=len(x), freq="D")
   data = pd.DataFrame({"x": x, "y": y, "z": z}, index=dates)

   qc = SaQC(data)

.. doctest:: python

   >>> qc.data  #doctest:+NORMALIZE_WHITESPACE
                x |               y |              z | 
   ============== | =============== | ============== | 
   2020-01-01  12 | 2020-01-01    2 | 2020-01-01  34 | 
   2020-01-02  87 | 2020-01-02   12 | 2020-01-02  23 | 
   2020-01-03  45 | 2020-01-03   33 | 2020-01-03  89 | 
   2020-01-04  31 | 2020-01-04  133 | 2020-01-04  56 | 
   2020-01-05  18 | 2020-01-05    8 | 2020-01-05   5 | 
   2020-01-06  99 | 2020-01-06   33 | 2020-01-06   1 | 


Simple constraints
~~~~~~~~~~~~~~~~~~

**Task**: Flag all values of ``x`` where ``x`` is smaller than 30

.. tabs::

   .. tab:: API

     .. testcode:: python

        qc1 = qc.flagGeneric(field="x", func=lambda x: x < 30)

     .. doctest:: python
        
        >>> qc1.flags  #doctest:+NORMALIZE_WHITESPACE
                        x |               y |               z | 
        ================= | =============== | =============== | 
        2020-01-01  255.0 | 2020-01-01 -inf | 2020-01-01 -inf | 
        2020-01-02   -inf | 2020-01-02 -inf | 2020-01-02 -inf | 
        2020-01-03   -inf | 2020-01-03 -inf | 2020-01-03 -inf | 
        2020-01-04   -inf | 2020-01-04 -inf | 2020-01-04 -inf | 
        2020-01-05  255.0 | 2020-01-05 -inf | 2020-01-05 -inf | 
        2020-01-06   -inf | 2020-01-06 -inf | 2020-01-06 -inf | 

   .. tab:: Configuration

     .. code-block::

        varname ; test                    
        #-------;------------------------
        x       ; flagGeneric(func=x < 30)

     .. doctest:: python
        :hide:

        >>> tmp = fromConfig(
        ...     writeIO(
        ...         """
        ...         varname ; test                    
        ...         #-------;------------------------
        ...         x       ; flagGeneric(func=x < 30)
        ...         """
        ...     ),
        ...     data
        ... )
        >>> (tmp.flags == qc1.flags).all(axis=None) #doctest:+NORMALIZE_WHITESPACE
        True


As to be expected, the usual `comparison operators`_ are supported.


Cross variable constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~
**Task**: Flag all values of ``x`` where ``y`` is larger than 30

.. tabs::

   .. tab:: API

     .. testcode:: python
        
        qc2 = qc.flagGeneric(field="y", target="x", func=lambda y: y > 30)

     .. doctest:: python
        
        >>> qc2.flags  #doctest:+NORMALIZE_WHITESPACE
                        x |               y |               z | 
        ================= | =============== | =============== | 
        2020-01-01   -inf | 2020-01-01 -inf | 2020-01-01 -inf | 
        2020-01-02   -inf | 2020-01-02 -inf | 2020-01-02 -inf | 
        2020-01-03  255.0 | 2020-01-03 -inf | 2020-01-03 -inf | 
        2020-01-04  255.0 | 2020-01-04 -inf | 2020-01-04 -inf | 
        2020-01-05   -inf | 2020-01-05 -inf | 2020-01-05 -inf | 
        2020-01-06  255.0 | 2020-01-06 -inf | 2020-01-06 -inf | 

     We introduce another selection parameter, namely ``target``. While ``field`` is still used to select
     a variable from the dataset, which is subsequently passed to the given function ``func``, ``target`` names the
     variable to which SaQC writes the produced flags.

   .. tab:: Configuration

     .. code-block::

        varname ; test                    
        #-------;------------------------------------
        x       ; flagGeneric(field="y", func=y > 30)


     Here the value in the first config column acts as the ``target``, while ``field`` needs to be given
     as function argument. In case ``field`` is not explicitly given, the first column acts as both,
     ``field`` and ``target``.

     .. doctest:: python
        :hide:

        >>> tmp = fromConfig(
        ...     writeIO(
        ...         """
        ...         varname ; test                    
        ...         #-------;------------------------------------
        ...         x       ; flagGeneric(field="y", func=y > 30)
        ...         """
        ...     ),
        ...     data
        ... )
        >>> (tmp.flags == qc2.flags).all(axis=None) #doctest:+NORMALIZE_WHITESPACE
        True


Multiple cross variable constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
**Task**: Flag all values of ``x`` where ``y`` is larger than 30 and ``z`` is smaller than 50:

In order to pass multiple variables to ``func``, we need to also specify multiple ``field`` elements. 
Note: to combine boolean expressions using one the available `logical operators`_, they single expressions
need to be put in parentheses.

.. tabs::

   .. tab:: API

     .. testcode:: python
        
        qc3 = qc.flagGeneric(field=["y", "z"], target="x", func=lambda y, z: (y > 30) & (z < 50))

     .. doctest:: python
        
        >>> qc3.flags  #doctest:+NORMALIZE_WHITESPACE
                        x |               y |               z | 
        ================= | =============== | =============== | 
        2020-01-01   -inf | 2020-01-01 -inf | 2020-01-01 -inf | 
        2020-01-02   -inf | 2020-01-02 -inf | 2020-01-02 -inf | 
        2020-01-03   -inf | 2020-01-03 -inf | 2020-01-03 -inf | 
        2020-01-04   -inf | 2020-01-04 -inf | 2020-01-04 -inf | 
        2020-01-05   -inf | 2020-01-05 -inf | 2020-01-05 -inf | 
        2020-01-06  255.0 | 2020-01-06 -inf | 2020-01-06 -inf | 

     The mapping
     from ``field`` to the ``lambda`` function parameters is positional and not bound to names. That means
     that the function parameters can be named arbitrarily.
   .. tab:: Configuration

     .. code-block::

        varname ; test                    
        #-------;--------------------------------------------------------
        x       ; flagGeneric(field=["y", "z"], func=(y > 30) & (z < 50))


     Here the value in the first config column acts as the ``target``, while ``field`` needs to be given
     as a function argument. In case ``field`` is not explicitly given, the first column acts as both,
     ``field`` and ``target``.
     The mapping from ``field`` to the names used in ``func`` is positional, i.e. the first value in ``field``
     is mapped to the first variable found in ``func``.

     .. doctest:: python
        :hide:

        >>> tmp = fromConfig(
        ...     writeIO(
        ...         """
        ...         varname ; test                    
        ...         #-------;--------------------------------------------------------
        ...         x       ; flagGeneric(field=["y", "z"], func=(y > 30) & (z < 50))
        ...         """
        ...     ),
        ...     data
        ... )
        >>> (tmp.flags == qc3.flags).all(axis=None) #doctest:+NORMALIZE_WHITESPACE
        True


Arithmetics
~~~~~~~~~~~
**Task**: Flag all values of ``x``, that exceed the arithmetic mean of ``y`` and ``z``

.. tabs::

   .. tab:: API

      .. testcode:: python

         qc4 = qc.flagGeneric(field=["x", "y", "z"], target="x", func=lambda x, y, z: x > (y + z)/2)

      .. doctest:: python

         >>> qc4.flags  #doctest:+NORMALIZE_WHITESPACE
                         x |               y |               z | 
         ================= | =============== | =============== | 
         2020-01-01   -inf | 2020-01-01 -inf | 2020-01-01 -inf | 
         2020-01-02  255.0 | 2020-01-02 -inf | 2020-01-02 -inf | 
         2020-01-03   -inf | 2020-01-03 -inf | 2020-01-03 -inf | 
         2020-01-04   -inf | 2020-01-04 -inf | 2020-01-04 -inf | 
         2020-01-05  255.0 | 2020-01-05 -inf | 2020-01-05 -inf | 
         2020-01-06  255.0 | 2020-01-06 -inf | 2020-01-06 -inf |               
     
   .. tab:: Configuration

      .. code-block::

         varname ; test
         #-------;-------------------------------------------------------
         x       ; flagGeneric(field=["x", "y", "z"], func=x > (y + z)/2)

      
      :py:meth:`flagGeneric <saqq.SaQC.flagGeneric>` supports the usual `arithmetic operators`_.

     .. doctest:: python
        :hide:

        >>> tmp = fromConfig(
        ...     writeIO(
        ...         """
        ...         varname ; test
        ...         #-------;-------------------------------------------------------
        ...         x       ; flagGeneric(field=["x", "y", "z"], func=x > (y + z)/2)
        ...         """
        ...     ),
        ...     data
        ... )
        >>> (tmp.flags == qc4.flags).all(axis=None) #doctest:+NORMALIZE_WHITESPACE
        True



Special functions
~~~~~~~~~~~~~~~~~

**Task**: Flag all values of ``x``, that exceed 2 standard deviations of ``z``.

.. tabs::

   .. tab:: API

      .. testcode:: python

         qc5 = qc.flagGeneric(field=["x", "z"], target="x", func=lambda x, z: x > np.std(z) * 2)

      .. doctest:: python

         >>> qc5.flags  #doctest:+NORMALIZE_WHITESPACE
                         x |               y |               z | 
         ================= | =============== | =============== | 
         2020-01-01   -inf | 2020-01-01 -inf | 2020-01-01 -inf | 
         2020-01-02  255.0 | 2020-01-02 -inf | 2020-01-02 -inf | 
         2020-01-03   -inf | 2020-01-03 -inf | 2020-01-03 -inf | 
         2020-01-04   -inf | 2020-01-04 -inf | 2020-01-04 -inf | 
         2020-01-05   -inf | 2020-01-05 -inf | 2020-01-05 -inf | 
         2020-01-06  255.0 | 2020-01-06 -inf | 2020-01-06 -inf | 
                
      The selected variables are passed to ``func`` as arguments of type ``pd.Series``, so all functions
      accepting such an argument can be used in generic expressions.
     
   .. tab:: Configuration

      .. code-block::

         varname ; test
         #-------;---------------------------------------------------
         x       ; flagGeneric(field=["x", "z"], func=x > std(z) * 2)

      In configurations files, the number of available mathematical functions is more restricted. Instead
      of basically all functions accepting array-like inputs, only certain built in
      `<mathematical functions>`_ can be used.

      .. doctest:: python
        :hide:

        >>> tmp = fromConfig(
        ...     writeIO(
        ...         """
        ...         varname ; test
        ...         #-------;---------------------------------------------------
        ...         x       ; flagGeneric(field=["x", "z"], func=x > std(z) * 2)
        ...         """
        ...     ),
        ...     data
        ... )
        >>> (tmp.flags == qc5.flags).all(axis=None) #doctest:+NORMALIZE_WHITESPACE
        True



**Task**: Flag all values of ``x`` where ``y`` is flagged.

.. tabs::

   .. tab:: API

      .. testcode:: python

         qc6 = (qc
                .flagRange(field="y", min=10, max=60)
                .flagGeneric(field="y", target="x", func=lambda y: isflagged(y)))

      .. doctest:: python

         >>> qc6.flags  #doctest:+NORMALIZE_WHITESPACE
                         x |                 y |               z | 
         ================= | ================= | =============== | 
         2020-01-01  255.0 | 2020-01-01  255.0 | 2020-01-01 -inf | 
         2020-01-02   -inf | 2020-01-02   -inf | 2020-01-02 -inf | 
         2020-01-03   -inf | 2020-01-03   -inf | 2020-01-03 -inf | 
         2020-01-04  255.0 | 2020-01-04  255.0 | 2020-01-04 -inf | 
         2020-01-05  255.0 | 2020-01-05  255.0 | 2020-01-05 -inf | 
         2020-01-06   -inf | 2020-01-06   -inf | 2020-01-06 -inf |     

   .. tab:: Configuration

      .. code-block::

         varname ; test
         #-------;------------------------------------------
         y       ; flagRange(min=10, max=60)
         x       ; flagGeneric(field="y", func=isflagged(y))

      .. doctest:: python
        :hide:

        >>> tmp = fromConfig(
        ...     writeIO(
        ...         """
        ...         varname ; test
        ...         #-------;------------------------------------------
        ...         y       ; flagRange(min=10, max=60)
        ...         x       ; flagGeneric(field="y", func=isflagged(y))
        ...         """
        ...     ),
        ...     data
        ... )
        >>> (tmp.flags == qc6.flags).all(axis=None) #doctest:+NORMALIZE_WHITESPACE
        True



A real world example
~~~~~~~~~~~~~~~~~~~~

Let's consider the following dataset:

.. testcode:: python

   from saqc import SaQC

   meas = np.array([3.56, 4.7, 0.1, 3.62])
   fan = np.array([1, 0, 1, 1])
   volt = np.array([12.1, 12.0, 11.5, 12.1])

   dates = pd.to_datetime(["2018-06-01 12:00", "2018-06-01 12:10", "2018-06-01 12:20", "2018-06-01 12:30"])
   data = pd.DataFrame({"meas": meas, "fan": fan, "volt": volt}, index=dates)

   qc = SaQC(data)

.. doctest:: python

   >>> qc.data  #doctest:+NORMALIZE_WHITESPACE
                        meas |                     fan |                      volt | 
   ========================= | ======================= | ========================= | 
   2018-06-01 12:00:00  3.56 | 2018-06-01 12:00:00   1 | 2018-06-01 12:00:00  12.1 | 
   2018-06-01 12:10:00  4.70 | 2018-06-01 12:10:00   0 | 2018-06-01 12:10:00  12.0 | 
   2018-06-01 12:20:00  0.10 | 2018-06-01 12:20:00   1 | 2018-06-01 12:20:00  11.5 | 
   2018-06-01 12:30:00  3.62 | 2018-06-01 12:30:00   1 | 2018-06-01 12:30:00  12.1 | 


**Task**: Flag ``meas`` where ``fan`` equals 0 and ``volt`` is lower than ``12.0``.

**Configuration file**: There are various options. We can directly implement the condition as follows:

.. tabs::

   .. tab:: API

      .. testcode:: python

         qc7 = qc.flagGeneric(field=["fan", "volt"], target="meas", func=lambda x, y: (x == 0) | (y < 12))

      .. doctest:: python

         >>> qc7.flags  #doctest:+NORMALIZE_WHITESPACE
                               meas |                      fan |                     volt | 
         ========================== | ======================== | ======================== | 
         2018-06-01 12:00:00   -inf | 2018-06-01 12:00:00 -inf | 2018-06-01 12:00:00 -inf | 
         2018-06-01 12:10:00  255.0 | 2018-06-01 12:10:00 -inf | 2018-06-01 12:10:00 -inf | 
         2018-06-01 12:20:00  255.0 | 2018-06-01 12:20:00 -inf | 2018-06-01 12:20:00 -inf | 
         2018-06-01 12:30:00   -inf | 2018-06-01 12:30:00 -inf | 2018-06-01 12:30:00 -inf |

   .. tab:: Configuration

      .. code-block::

         varname ; test
         #-------;---------------------------------------------------------------
         meas    ; flagGeneric(field=["fan", "volt"], func=(x == 0) | (y < 12.0))


      .. doctest:: python
        :hide:

        >>> tmp = fromConfig(
        ...     writeIO(
        ...         """
        ...         varname ; test
        ...         #-------;---------------------------------------------------------------
        ...         meas    ; flagGeneric(field=["fan", "volt"], func=(x == 0) | (y < 12.0))
        ...         """
        ...     ),
        ...     data
        ... )
        >>> (tmp.flags == qc7.flags).all(axis=None) #doctest:+NORMALIZE_WHITESPACE
        True


But we could also quality check our independent variables first and than leverage this information later on:

.. tabs::

   .. tab:: API

      .. testcode:: python

         qc8 = (qc
                .flagMissing(".*", regex=True)
                .flagGeneric(field="fan", func=lambda x: x == 0)
                .flagGeneric(field="volt", func=lambda x: x < 12)
                .flagGeneric(field=["fan", "volt"], target="meas", func=lambda x, y: isflagged(x) | isflagged(y)))

      .. doctest:: python

         >>> qc8.flags  #doctest:+NORMALIZE_WHITESPACE
                               meas |                        fan |                       volt | 
         ========================== | ========================== | ========================== | 
         2018-06-01 12:00:00   -inf | 2018-06-01 12:00:00   -inf | 2018-06-01 12:00:00   -inf | 
         2018-06-01 12:10:00  255.0 | 2018-06-01 12:10:00  255.0 | 2018-06-01 12:10:00   -inf | 
         2018-06-01 12:20:00  255.0 | 2018-06-01 12:20:00   -inf | 2018-06-01 12:20:00  255.0 | 
         2018-06-01 12:30:00   -inf | 2018-06-01 12:30:00   -inf | 2018-06-01 12:30:00   -inf |

   .. tab:: Configuration

      .. code-block::

         varname ; test
         #-------;--------------------------------------------------------------------------
         '.*'    ; flagMissing()
         fan     ; flagGeneric(func=fan == 0)
         volt    ; flagGeneric(func=volt < 12.0)
         meas    ; flagGeneric(field=["fan", "volt"], func=isflagged(fan) | isflagged(volt))

      .. doctest:: python
        :hide:

        >>> tmp = fromConfig(
        ...     writeIO(
        ...         """
        ...         varname ; test
        ...         #-------;--------------------------------------------------------------------------
        ...         '.*'    ; flagMissing()
        ...         fan     ; flagGeneric(func=fan == 0)
        ...         volt    ; flagGeneric(func=volt < 12.0)
        ...         meas    ; flagGeneric(field=["fan", "volt"], func=isflagged(fan) | isflagged(volt))
        ...         """
        ...     ),
        ...     data
        ... )
        >>> (tmp.flags == qc8.flags).all(axis=None) #doctest:+NORMALIZE_WHITESPACE
        True


Generic Processing
------------------

Generic processing functions provide a way to evaluate mathematical operations 
and functions on the variables of a given dataset.

Why
^^^

In many real-world use cases, quality control is embedded into a larger data 
processing pipeline. It is not unusual to even have certain processing 
requirements as a part of the quality control itself. Generic processing 
functions make it easy to enrich a dataset through the evaluation of a
given expression.

Generic Processing - Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The basic signature looks like that:

.. code-block:: sh

   processGeneric(field, func=<expression>)

where ``<expression>`` is either a callable (Python API) or an expression composed of the
`supported constructs`_ (Configuration File).

Example
^^^^^^^

Let's use :py:meth:`flagGeneric <saqq.SaQC.processGeneric>` to calculate the mean value of several
variables in a given dataset. We start with dummy data again:

.. testcode:: python
              
   from saqc import SaQC

   x = np.array([12, 87, 45, 31, 18, 99])
   y = np.array([2, 12, 33, 133, 8, 33])
   z = np.array([34, 23, 89, 56, 5, 1])

   dates = pd.date_range(start="2020-01-01", periods=len(x), freq="D")
   data = pd.DataFrame({"x": x, "y": y, "z": z}, index=dates)

   qc = SaQC(data)

.. tabs::

   .. tab:: API

     .. testcode:: python

        qc1 = qc.processGeneric(
                   field=["x", "y", "z"],
                   target="mean",
                   func=lambda x, y, z: (x + y + z) / 2
        )

     .. doctest:: python
        
        >>> qc1.data  #doctest:+NORMALIZE_WHITESPACE
                     x |               y |              z |              mean | 
        ============== | =============== | ============== | ================= | 
        2020-01-01  12 | 2020-01-01    2 | 2020-01-01  34 | 2020-01-01   24.0 | 
        2020-01-02  87 | 2020-01-02   12 | 2020-01-02  23 | 2020-01-02   61.0 | 
        2020-01-03  45 | 2020-01-03   33 | 2020-01-03  89 | 2020-01-03   83.5 | 
        2020-01-04  31 | 2020-01-04  133 | 2020-01-04  56 | 2020-01-04  110.0 | 
        2020-01-05  18 | 2020-01-05    8 | 2020-01-05   5 | 2020-01-05   15.5 | 
        2020-01-06  99 | 2020-01-06   33 | 2020-01-06   1 | 2020-01-06   66.5 |

     The call to :py:meth:`flagGeneric <saqq.SaQC.processGeneric>` added the new variable ``mean``
     to the dataset.

   .. tab:: Configuration

     .. code-block::

        varname ; test                    
        #-------;------------------------------------------------------
        mean    ; processGeneric(field=["x", "y", "z"], func=(x+y+z)/2)

     .. doctest:: python
        :hide:

        >>> tmp = fromConfig(
        ...     writeIO(
        ...         """
        ...         varname ; test                    
        ...         #-------;------------------------------------------------------
        ...         mean    ; processGeneric(field=["x", "y", "z"], func=(x+y+z)/2)
        ...         """
        ...     ),
        ...     data
        ... )
        >>> (tmp.data == qc1.data).all(axis=None) #doctest:+NORMALIZE_WHITESPACE
        True


Supported constructs
--------------------

Operators
^^^^^^^^^

Comparison Operators
~~~~~~~~~~~~~~~~~~~~

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


Logical operators
~~~~~~~~~~~~~~~~~

The bitwise operators act as logical operators in comparison chains

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


Arithmetic Operators
~~~~~~~~~~~~~~~~~~~~

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


Functions
^^^^^^^^^

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
   * - ``abs``
     - Pointwise absolute Value Function.
   * - ``max``
     - Maximum Value Function. Ignores NaN.
   * - ``min``
     - Minimum Value Function. Ignores NaN.
   * - ``mean``
     - Mean Value Function. Ignores NaN.
   * - ``sum``
     - Summation. Ignores NaN.
   * - ``len``
     - Standart Deviation. Ignores NaN.
   * - ``exp``
     - Pointwise Exponential.
   * - ``log``
     - Pointwise Logarithm.
   * - ``nanLog``
     - Logarithm, returning NaN for zero input, instead of -inf.
   * - ``std``
     - Standart Deviation. Ignores NaN.
   * - ``var``
     - Variance. Ignores NaN.
   * - ``median``
     - Median. Ignores NaN.
   * - ``count``
     - Count Number of values. Ignores NaNs.
   * - ``id``
     - Identity.
   * - ``diff``
     - Returns a Series` diff.
   * - ``scale``
     - Scales data to [0,1] Interval.
   * - ``zScore``
     - Standardize with Standart Deviation.
   * - ``madScore``
     - Standardize with Median and MAD.
   * - ``iqsScore``
     - Standardize with Median and inter quantile range.


Miscellaneous Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - ``isflagged``
     - Pointwise, checks if a value is flagged
   * - ``len``
     - Returns the length of passed series
