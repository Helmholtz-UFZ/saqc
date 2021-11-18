Python API
==========

General structure
-----------------

The Python API of SaQC consists of three distinct components:
1. the core class ``SaQC``
2. a number of flagging schemes
3. a collection of functions

One and two are implemented as distinct classes, the core object is called ``SaQC`` and we currently
provide three flagging schemes, namely:

1. ``FloatTranslators``: Provides the quality flags ``-np.inf`` and ``[0..255]``. ``-np.inf`` denotes the
   absence of quality flags and ``255`` denotes, that the associated data value is considered to be bad.
   the absence of a flags, ``1`` the presence of flag (i.e. one of the tests provided a positive result)
   and ``0`` acts as an indicator for an tested-to-be-not-bad value.
2. ``SimpleTranslator``: Provides three distinct quality labels, namely ``UNFLAGGED``, ``BAD``, ``OKAY``.
3. ``DmpTranslator``: Provides the four distinct flags ``NIL``, ``OK``, ``DOUBTFUL``, ``BAD``, whereas each
   flag is extended by information about the generating function and optional comments

The third component, the actual test functions, appear as methods of ``SaQC``.


Getting started - Put something in
----------------------------------

The definition of a ``SaQC`` test suite starts with some data as a ``pandas.DataFrame`` and the selection
of an appropriate flagging scheme. For reasons of simplicity, we'll use the ``SimpleTranslator`` throughout
the following examples. However, as the flagging schemes are interchangable, replacing the ``SimpleTranslator``
with something more elaborate, is in fact a one line change. So let's start with:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from saqc import SaQC

   # we need some dummy data
   values = np.random.randint(low=0, high=100, size=100)
   dates = pd.date_range(start="2020-01-01", periods=len(values), freq="D")
   data = pd.DataFrame({"a": values}, index=dates)
   # let's insert some constant values ...
   data.iloc[30:40] = values.mean()
   # ... and an outlier
   data.iloc[70] = 175

   # initialize saqc
   qc = SaQC(data=data, scheme="simple")


Moving on - Quality control your data
-------------------------------------

The ``qc`` variable now serves as the base for all further processing steps. As mentioned above, all
available functions appear as methods of the ``SaQC``  class, so we can add a tests to our suite with:

.. code-block:: python

   qc.flagRange("a", min=20, max=80)

``flagRange`` is the easiest of all functions and simply marks all values smaller than ``min`` and larger
than ``max``. This feature by itself wouldn't be worth the trouble of getting into ``SaQC``, but it serves
as a simple example. All functions expect the name of a column in ``data`` as the first positional argument
(called ``field``). The function ``flagRange`` (like all other functions for that matter) is then called on
the given ``field`` (only).

Each call to a ``SaQC`` method returns a new object (all itermediate objects share the main internal data
structures, so we only create shallow copies). Setting up more complex quality control suites is therefore
simply a matter of method chaining. 

.. code-block:: python
   # execute some tests
   qc = (qc
         .flagConstants("a", thresh=0.1, window="4D")
         .flagByGrubbs("a", window="10D")
         .flagRange("a", min=20, max=80))


Getting done - Pull something out
---------------------------------

``saqc`` is eagerly evaluated, i.e. the results of all method calls are available as soon as they return. As
we have seen above, calling quality checks does however not immediately return the produces data and the
associated flags, but rather an new ``SaQC`` object. The actual execution products are accessible through a
number of different attributes, of which you likely might want to use the following:

.. code-block:: python

   # retrieve the data as a pandas.DataFrame
   qc.data

   # retrieve the flags as a pandas.DataFrame
   qc.flags


Putting it together - The complete workflow
-------------------------------------------
The snippet below provides you with a compete example from the things we have seen so far.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from saqc import SaQC

   # we need some dummy data
   values = np.random.randint(low=0, high=100, size=100)
   dates = pd.date_range(start="2020-01-01", periods=len(values), freq="D")
   data = pd.DataFrame({"a": values}, index=dates)
   # let's insert some constant values ...
   data.iloc[30:40] = values.mean()
   # ... and an outlier
   data.iloc[70] = 175

   # initialize saqc
   qc = SaQC(data=data, scheme="simple")

   # execute some tests
   qc = (qc
         .flagConstants("a", thresh=0.1, window="4D")
         .flagByGrubbs("a", window="10D")
         .flagRange("a", min=20, max=80))

   # retrieve the data as a pandas.DataFrame
   qc.data

   # retrieve the flags as a pandas.DataFrame
   qc.flags



Can I get something visual, please?
-----------------------------------

Yes, you can. We provide an elaborated plotting method to generate and show or write matplotlib figures.
Building on the example :ref:`above <getting_started/TutorialAPI:putting it together - a complete workflow>`
the calling the method ``qc.plot("a")`` will generate a plot like the following:

.. image:: /ressources/images/tutorial_api_1.png
