.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Documentation Guide
===================

We document our code via docstrings in numpy-style. 
Features, install and usage instructions and other more text intense stuff, 
is written in extra documents. 
The documents and the docstrings then are collected and rendered using `sphinx <https://www.sphinx-doc.org/>`_. 

Documentation Strings
---------------------


* Write docstrings for all public modules, functions, classes, and methods.
  Docstrings are not necessary for non-public methods,
  but you should have a comment that describes what the method does.
  This comment should appear after the def line.
  [\ `PEP8 <https://www.python.org/dev/peps/pep-0008/#documentation-strings>`_\ ]

* Note that most importantly, the ``"""`` that ends a multiline docstring should be on a line by itself [\ `PEP8 <https://www.python.org/dev/peps/pep-0008/#documentation-strings>`_\ ] :

  .. code-block:: python

       """Return a foobang

       Optional plotz says to frobnicate the bizbaz first.
       """

* For one liner docstrings, please keep the closing ``"""`` on the same line.
  [\ `PEP8 <https://www.python.org/dev/peps/pep-0008/#documentation-strings>`_\ ]

Pandas Style
^^^^^^^^^^^^

We use `Pandas-style <https://pandas.pydata.org/pandas-docs/stable/development/contributing_docstring.html>`_ docstrings:

Flagger, data, field, etc.
--------------------------

use this:

.. code-block:: py

   def foo(data, field, flagger):
       """
       data : dios.DictOfSeries
       A saqc-data object.

       field : str
       A field denoting a column in data.

       flagger : saqc.flagger.BaseFlagger
       A saqc-flagger object.
       """

IDE helper
^^^^^^^^^^

In pycharm one can activate autogeneration of numpy doc style like so:

#. ``File->Settings...``
#. ``Tools->Python Integrated Tools``
#. ``Docstrings->Docstring format``
#. Choose ``NumPy``

Docstring formatting pitfalls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Latex is included via:

  .. code-block::

     :math:`<latex_code>`

* Latex commands need to be signified with **double**   backlash! (``\\mu``)
* Nested lists need to be all of the same kind (either   numbered or marked - otherwise result is salad)
* List items covering several lines in the docstring have to be all aligned - (so, not only the superfluent ones, but ALL, including the first one - otherwise result is salad)
* Start of a list has to be seperated from preceding docstring code by *one blank line* - (otherwise list items get just chained in one line and result is salad)
* Most formatting signifiers are not allowed to start or end with a space.
* Do not include lines *only* containing two or more ``-`` signs, except it is the underscore line of the section heading (otherwise resulting html representation could be messed up)

hyperlinking docstrings
-----------------------

* Link code content/modules via python roles.
* Cite/link via the py domain roles. Link content ``bar``\ , that is registered to the API with the adress ``foo.bar`` and
  shall be represented by the name ``link_name``\ , via: 

  .. code-block::

     :py:role:`link_name <foo.bar>`

* Check out the *_api* folder in the `repository <https://git.ufz.de/rdm-software/saqc/-/tree/develop/sphinx-doc>`_ to get an
  overview of already registered paths. Most important may be:
* Constants are available via ``saqc.constants`` - for example:

  .. code-block::

     :py:const:`~saqc.constants.BAD`

* The ``~`` is a shorthand for hiding the module path and only displaying ``BAD``.
* Functions are available via the "simulated"  module ``Functions.saqc`` - for example:

  .. code-block::

     :py:func:`coolMethod <saqc.SaQC.flagRange>`


* The saqc object and/or its content is available via:

  .. code-block::

     :py:class:`saqc.SaQC`
     :py:meth:`saqc.SaQC.flagRange`


* The Flags object and/or its content is available via:

  .. code-block::

     :py:class:`saqc.Flags`

* You can add .rst files containing ``automodapi`` directives to the modulesAPI folder to make available more modules via pyroles
* The Environment table, including variables available via config files is available as restfile located in the environment folder. (Use include directive to include, or linking syntax to link it.

Integrating doctested code snippets
-----------------------------------

code-block
^^^^^^^^^^

If you want to ONLY RENDER code blocks, use the common `code-block` directive:

.. code-block:: rest

   .. code-block:: python

      a = 1
      b = 2
      a + b

This results in:

.. code-block:: python

      a = 1
      b = 2
      a + b

testcode
^^^^^^^^

If you want code to be RENDERED and TESTED, use the `testcode` directive. You can specify a group, where
assignments and imports will be stored to, and available for later `testcode` directives
of the same group and same document. Code is executed, and the doctest will fail, if execution causes an exception to be thrown.

.. code-block:: rest

   .. testcode:: group1

      a = 1
      b = 2
      a + b

This will be rendered as:

.. testcode:: group1

      a = 1
      b = 2
      a + b

Assignments (and imports) will be available in any other `testcode` directive, that has the same group assigned.
So the following wont fail, since ``a`` is known in ``group1``:

.. code-block:: rest

   .. testcode:: group1

      a - 4

testsetup
^^^^^^^^^

If you want to setup the example environment in a hidden manor, you can use the `testsetup` directive:

.. code-block:: rest

   .. testsetup:: group1

      import scipy

Will import scipy into the group environment, omitting rendering/display of the code.

testoutput
^^^^^^^^^^

If you want to additionally check the final *std_out* output of a `testcode` block, you can use
the `testoutput` directive:

.. code-block:: rest

   .. testcode:: group1

      a - 4

   .. testoutput::

      -3

This will be rendered as follows:

.. testcode:: group1

   a - 4

.. testoutput::

   -3

You can omit displaying of the testoutput, by adding the `hidden` option.

doctest
^^^^^^^

If you want to have code tested and rendered in the doctest style rendering,
(including *>>>*), you can use doctest syntax:

.. code-block:: rest

   >>> 1+1
   2

This will be rendered as:

>>> 1+1
2

It can be a little tricky, to match complexer std_out strings, like dios or DataFrames. There are some
doctest flags that can mitigate frustration:

#. NORMALIZE_WHITESPACE will map all whitespace/tab combos onto a single whitespace. Use like:

   .. code-block:: rest

      >>> 'ab  c' #doctest:+NORMALIZE_WHITESPACE
      'ab c'

#. ELLIPSIS will allow usage of the '...'-Wildcard in the expected output. (Usefull, if output contains unpredictable substrings, like memory adresses or filepaths

   .. code-block:: rest

      >>> 'abcdefg' #doctest:+ELLIPSIS
      'a...b'

#. SKIP skips the check (and execution of the line) all together. (usually used, if display is demanded, but testing would somehow be unstable, due to random/unpredictable components)

   .. code-block:: rest

      >>> time #doctest:+SKIP
      CPU times: user 5 µs, sys: 3 µs, total: 8 µs
      Wall time: 13.8 µs

   .. caution::
      Skipped lines are NOT tested! The execution of the line is skipped all together with the check against the
      expected output.

To assign a group to doctest snippets, use the more verbose `doctest` directive:

.. code-block:: rest

   .. doctest:: group1

      >>> time #doctest:+SKIP
      CPU times: user 5 µs, sys: 3 µs, total: 8 µs
      Wall time: 13.8 µs

Will be rendered, as:

.. doctest:: group1

      >>> time #doctest:+SKIP
      CPU times: user 5 µs, sys: 3 µs, total: 8 µs
      Wall time: 13.8 µs


Run doctest locally
-------------------

Since doctest checks guard the push to the develop branch, you might wish to chek if your local modification passes
all doctests beforehand.

There for go to the docs directory and run:

.. code-block::

   make test

To only run the doctests.
