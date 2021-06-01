# Documentation Guide

We document our code via docstrings in numpy-style. 
Features, install and usage instructions and other more text intense stuff, 
is written in extra documents. 
The documents and the docstrings then are collected and rendered using [sphinx](https://www.sphinx-doc.org/). 

 
## Documentation Strings


- Write docstrings for all public modules, functions, classes, and methods. 
    Docstrings are not necessary for non-public methods, 
    but you should have a comment that describes what the method does. 
    This comment should appear after the def line. 
    [[PEP8](https://www.python.org/dev/peps/pep-0008/#documentation-strings)]

- Note that most importantly, the `"""` that ends a multiline docstring should be on a line by itself [[PEP8](https://www.python.org/dev/peps/pep-0008/#documentation-strings)] :
    ```python
    """Return a foobang

    Optional plotz says to frobnicate the bizbaz first.
    """
    ```
    
- For one liner docstrings, please keep the closing `"""` on the same line. 
  [[PEP8](https://www.python.org/dev/peps/pep-0008/#documentation-strings)]

### Pandas Style

We use [Pandas-style](https://pandas.pydata.org/pandas-docs/stable/development/contributing_docstring.html) docstrings:



## Flagger, data, field, etc.

use this:
```py
def foo(data, field, flagger):
    """
    data : dios.DictOfSeries
	A saqc-data object.

    field : str
	A field denoting a column in data.

    flagger : saqc.flagger.BaseFlagger
	A saqc-flagger object.
    """
```


### IDE helper

In pycharm one can activate autogeneration of numpy doc style like so:
1. `File->Settings...`
2. `Tools->Python Integrated Tools`
3. `Docstrings->Docstring format`
4. Choose `NumPy`


### Docstring formatting pitfalls

* Latex is included via 
```
:math:`<latex_code>`
```
* Latex commands need to be signified with **double**   backlash! (``\\mu`` instead of ``\mu``)

* Nested lists need to be all of the same kind (either   numbered or marked - otherwise result is salad) 
* List items covering several lines in the docstring have to be all aligned - (so, not only the superfluent ones, but ALL, including the first one - otherwise result is salad)
* Start of a list has to be seperated from preceding docstring code by *one blank line* - (otherwise list items get just chained in one line and result is salad)
* Most formatting signifiers are not allowed to start or end with a space. (so no :math: \`1+1 \`, \` var2\`, \`\` a=1 \`\`, ...)
* Do not include lines *only* containing two or more `-` signs, except it is the underscore line of the section heading (otherwise resulting html representation could be messed up)

## hyperlinking docstrings

* Most straight forward way to make documented code content available / linkable, is, adding a rest file containing an
  automodapi directive to the folder `moduleAPIs` - check out the files it already contains as example.
  * adding ``.. automodapi:: foo.bar``, will make the module `foo.bar` and all its content `foo.bar.X` referable by the 
    module path.
    
* Cite/link via the py domain roles. Link content `bar`, that is registered to the API with the adress `foo.bar` and 
  shall be represented by the name `link_name`, via: 
```  
:py:role:`link_name <foo.bar>`
```    
* check out the *_api* folder in the [repository](https://git.ufz.de/rdm-software/saqc/-/tree/develop/sphinx-doc) to get an
  overview of already registered paths. Most important may be:
  
* constants are available via `saqc.constants` - for example:
``` 
:py:const:`~saqc.constants.BAD` 
```  

* Functions are available via the "fake"  module `Functions.saqc` - for example: 
  
``` 
:py:func:`saqc.flagRange <saqc.Functions.flagRange>` 
``` 
  
* The saqc object and/or its content is available via: 
  
```
:py:class: `saqc.SaQC` 
:py:meth: `saqc.SaQC.show` 
```   

## Linking to function categories

To link to a group of functions, you might generate a rest landing page for that link and add it to 
the `function_cats` folder. It already contains the landing pages for the :doc:`generic <../function_cats/generic>` 
functions and the :doc:`regularisation <../function_cats/regularisation>` functions.
Those can be linked via relative paths to the `function_cats` folder. From this file, located in `sphinx-doc/how_to_doc`,
the linking is realized by:

```python
:doc:`regularisation <../function_cats/regularistaion>`
:doc:`generic <../function_cats/generic>`
```

## Adding Markdown content to the Documentation

- By linking the markdown file "foo/bar.md", or any folder that contains markdown files directly, 
  you can trigger sphinx - `recommonmark`, which is fine for not-too complex markdown documents. 
  
* Especially, if you have multiple markdown files that are mutually linked and/or, contain tables of certain fencieness (tables with figures),
  you will have to take some minor extra steps:
  
- You will have to gather all markdown files in subfolders of "sphinx-doc" directory (you can have multiple subfolders). 

- To include a folder named `foo` of markdown files in the documentation, or refer to content in `foo`, you will have 
  to append the folder name to the MDLIST variable in the Makefile:

- The markdown files must be in one of the subfolders listed in MDLIST - they cant be gathered in nested subfolders. 

- You can not link to sections in other markdown files, that contain the `-` character (sorry).

- The Section structure/ordering must be consistent in the ReST sence (otherwise they wont appear - thats also required if you use plain `recommonmark`

- You can link to ressources - like pictures and include them in the markdown, if the pictures are in (possibly another) folder in `\sphinx-doc` and the paths to this ressources are given relatively!

- You can include a markdown file in a rest document, by appending '_m2r' to the folder name when linking it path_wise. 
  So, to include the markdown file 'foo/bar.md' in a toc tree for example - you would do something like:

```python
.. toctree::
   :hidden:
   :maxdepth: 1

   foo_m2r/bar
```

## Linking ReST sources in markdown documentation

- If you want to hyperlink/include other sources from the sphinx documentation that are rest-files (and docstrings), 
  you will not be able to include them in a way, that they will appear in you markdown rendering. - however - there is 
  the posibillity to just include the respective rest directives (see directive/link [examples](#hyperlinking-docstrings)). 
  
- This will mess up your markdown code - meaning that you will have 
  those rest snippets flying around, but when the markdown file gets converted to the rest file and build into the 
  sphinx html build, the linked sources will be integrated properly. The syntax for linking rest sources is as 
  follows as follows:

- to include the link to the rest source `functions.rst` in the folder `foo`, under the name `bar`, you would need to insert: 
```python
:doc:`foo <rel_path/functions>`
```

- to link to a section with name `foo` in a rest source named `bumm.rst`, under the name `bar`, you would just insert: 
```
:ref:`bar <relative/path/from/sphinx/root/bumm:foo>`
``` 

- in that manner you might be able to smuggle most rest directives through into the resulting html build. Especially if you want to link to the docstrings of certain (domain specific) objects. Lets say you want to link to the *function* `saqc.funcs.flagRange` under the name `ranger` - you just include:

```
:py:func:`Ranger <saqc.funcs.flagRange>`
```

whereas the `:func:` part determines the role, the object is documented as. See [this page](https://www.sphinx-doc.org/en/master/#ref-role) for an overview of the available roles

