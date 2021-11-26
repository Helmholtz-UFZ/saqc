
Documentation Guide
===================

We document our code via docstrings in numpy-style. 
Features, install and usage instructions and other more text intense stuff, 
is written in extra documents. 
The documents and the docstrings then are collected and rendered using `sphinx <https://www.sphinx-doc.org/>`_. 

Documentation Strings
---------------------


* 
  Write docstrings for all public modules, functions, classes, and methods. 
    Docstrings are not necessary for non-public methods, 
    but you should have a comment that describes what the method does. 
    This comment should appear after the def line. 
    [\ `PEP8 <https://www.python.org/dev/peps/pep-0008/#documentation-strings>`_\ ]

* 
  Note that most importantly, the ``"""`` that ends a multiline docstring should be on a line by itself [\ `PEP8 <https://www.python.org/dev/peps/pep-0008/#documentation-strings>`_\ ] :

  .. code-block:: python

       """Return a foobang

       Optional plotz says to frobnicate the bizbaz first.
       """

* 
  For one liner docstrings, please keep the closing ``"""`` on the same line. 
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


* Latex is included via 
  .. code-block::

     :math:`<latex_code>`

* 
  Latex commands need to be signified with **double**   backlash! (\ ``\\mu`` instead of ``\mu``\ )

* 
  Nested lists need to be all of the same kind (either   numbered or marked - otherwise result is salad) 

* List items covering several lines in the docstring have to be all aligned - (so, not only the superfluent ones, but ALL, including the first one - otherwise result is salad)
* Start of a list has to be seperated from preceding docstring code by *one blank line* - (otherwise list items get just chained in one line and result is salad)
* Most formatting signifiers are not allowed to start or end with a space. (so no :math: `1+1 `, ` var2`, `` a=1 ``, ...)
* Do not include lines *only* containing two or more ``-`` signs, except it is the underscore line of the section heading (otherwise resulting html representation could be messed up)

hyperlinking docstrings
-----------------------


* 
  Link code content/modules via python roles.

* 
  Cite/link via the py domain roles. Link content ``bar``\ , that is registered to the API with the adress ``foo.bar`` and 
  shall be represented by the name ``link_name``\ , via: 

  .. code-block::

     :py:role:`link_name <foo.bar>`

* 
  check out the *_api* folder in the `repository <https://git.ufz.de/rdm-software/saqc/-/tree/develop/sphinx-doc>`_ to get an
  overview of already registered paths. Most important may be:

* 
  constants are available via ``saqc.constants`` - for example:

  .. code-block::

     :py:const:`~saqc.constants.BAD`

* 
  the ``~`` is a shorthand for hiding the module path and only displaying ``BAD``.

* 
  Functions are available via the "simulated"  module ``Functions.saqc`` - for example: 

.. code-block::

   :py:func:`saqc.flagRange <saqc.Functions.flagRange>`


* Modules are available via the "simulated"  package ``Functions.`` - for example: 

.. code-block::

   :py:mod:`generic <Functions.generic>`


* The saqc object and/or its content is available via: 

.. code-block::

   :py:class:`saqc.SaQC` 
   :py:meth:`saqc.SaQC.getResults`


* The Flags object and/or its content is available via: 

.. code-block::

   :py:class:`saqc.Flags`


* 
  you can add .rst files containing ``automodapi`` directives to the modulesAPI folder to make available more modules via pyroles

* 
  the Environment table, including variables available via config files is available as restfile located in the environment folder. (Use include directive to include, or linking syntax to link it.

Adding Markdown content to the Documentation
--------------------------------------------


* 
  By linking the markdown file "foo/bar.md", or any folder that contains markdown files directly, 
  you can trigger sphinx - ``recommonmark``\ , which is fine for not-too complex markdown documents. 

* 
  Especially, if you have multiple markdown files that are mutually linked and/or, contain tables of certain fencieness (tables with figures),
  you will have to take some minor extra steps:

* 
  You will have to gather all markdown files in subfolders of "sphinx-doc" directory (you can have multiple subfolders). 

* 
  To include a folder named ``foo`` of markdown files in the documentation, or refer to content in ``foo``\ , you will have 
  to append the folder name to the MDLIST variable in the Makefile:

* 
  The markdown files must be in one of the subfolders listed in MDLIST - they cant be gathered in nested subfolders. 

* 
  You can not link to sections in other markdown files, that contain the ``-`` character (sorry).

* 
  The Section structure/ordering must be consistent in the ReST sence (otherwise they wont appear - thats also required if you use plain ``recommonmark``

* 
  You can link to ressources - like pictures and include them in the markdown, if the pictures are in (possibly another) folder in ``\sphinx-doc`` and the paths to this ressources are given relatively!

* 
  You can include a markdown file in a rest document, by appending '_m2r' to the folder name when linking it path_wise. 
  So, to include the markdown file 'foo/bar.md' in a toc tree for example - you would do something like:

* 
  the Environment table, including variables availbe via config files is available as restfile located in the environment folder. (Use include directive to include, or linking syntax to link it.)

.. code-block:: python

   .. toctree::
      :hidden:
      :maxdepth: 1

      foo_m2r/bar

Linking ReST sources in markdown documentation
----------------------------------------------


* 
  If you want to hyperlink/include other sources from the sphinx documentation that are rest-files (and docstrings), 
  you will not be able to include them in a way, that they will appear in you markdown rendering. - however - there is 
  the posibillity to just include the respective rest directives (see directive/link :ref:`examples <how_to_doc/HowToDoc:hyperlinking docstrings>`\ ). 

* 
  This will mess up your markdown code - meaning that you will have 
  those rest snippets flying around, but when the markdown file gets converted to the rest file and build into the 
  sphinx html build, the linked sources will be integrated properly. The syntax for linking rest sources is as 
  follows as follows:

* 
  to include the link to the rest source ``functions.rst`` in the folder ``foo``\ , under the name ``bar``\ , you would need to insert: 

  .. code-block::

     :doc:`foo <rel_path/functions>`

* 
  to link to a section with name ``foo`` in a rest source named ``bumm.rst``\ , under the name ``bar``\ , you would just insert: 

  .. code-block::

     :ref:`bar <relative/path/from/sphinx/root/bumm:foo>`

* 
  in that manner you might be able to smuggle most rest directives through into the resulting html build. Especially if you want to link to the docstrings of certain (domain specific) objects. Lets say you want to link to the *function* ``saqc.funcs.flagRange`` under the name ``ranger`` - you just include:

.. code-block::

   :py:func:`Ranger <saqc.funcs.flagRange>`

whereas the ``:func:`` part determines the role, the object is documented as. See `this page <https://www.sphinx-doc.org/en/master/#ref-role>`_ for an overview of the available roles
