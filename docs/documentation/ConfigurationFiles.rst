.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Configuration Files
===================

The behaviour of SaQC can be completely controlled by a text based configuration file.

Format
------

SaQC expects configuration files to be semicolon-separated text files with one header line.
Each row of the configuration file lists one variable and test function that is applied on
the given variable.

Header names
^^^^^^^^^^^^

The first line of every configuration file is dropped anyways, so feel free to use header
names to your liking


Test function notation
^^^^^^^^^^^^^^^^^^^^^^

The notation of test functions follows the function call notation of Python and
many other programming languages and looks like this:

.. code-block::

   flagRange(min=0, max=100)

Here the function ``flagRange`` is called and the values ``0`` and ``100`` are passed
to the parameters ``min`` and ``max`` respectively. As we value readablity
of the configuration more than conciseness of the extension language, only
keyword arguments are supported. That means that the notation ``flagRange(0, 100)``
is not a valid replacement for the above example.

Examples
--------

Every row lists one test per variable. If you want to call multiple tests on
a specific variable (and you probably want to), list them in separate rows:

.. code-block::

   varname | test
   #-------|----------------------------------
   x       | flagMissing()
   x       | flagRange(min=0, max=100)
   x       | flagConstants(window="3h")
   y       | flagRange(min=-10, max=40)


Regular Expressions in ``varname`` column
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some of the tests (e.g. checks for missing values, range tests or interpolation
functions) are very likely to be used on all or at least several variables of
the processed dataset. As it becomes quite cumbersome to list all these
variables seperately, only to call the same functions with the same
parameters, SaQC supports regular expressions on variables. To mark a given
variable name as a regular expression, it needs to be quoted with ``'`` or ``"``

.. code-block::

   varname    ; test
   #----------;------------------------------
   '.*'       ; harm_shift2Grid(freq="15Min")
   '(x | y)' ; flagMissing()
