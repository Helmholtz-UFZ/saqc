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

* Latex is included via :math:\`latex_code\`
  
  * note, the backticks surrounding the actual code
  * Latex commands need to be signified with **double**    backlash! (``\\mu`` instead of ``\mu``)

* Nested lists need to be all of the same kind (either   numbered or marked - otherwise result is salad) 
* List items covering several lines in the docstring have to be all aligned - (so, not only the superfluent ones, but ALL, including the first one - otherwise result is salad)
* Start of a list has to be seperated from preceding docstring code by *one blank line* - (otherwise list items get just chained in one line and result is salad)
* Most formatting signifiers are not allowed to start or end with a space. (so no :math: \`1+1 \`, \` var2\`, \`\` a=1 \`\`, ...)
* Do not include lines *only* containing two or more `-` signs, except it is the underscore line of the section heading (otherwise resulting html representation could be messed up)

## Adding Markdown content to the Documentation

* If you generate cookbooks and/or tutorials in markdown and want them to be integrated in the sphinx doc - there are some obstaclish thingies to care for

- You will have to gather all markdown files in subfolders of "sphinx-doc". 

- To include a folder named 'foo_md' of markdown files in the documentation, you will have to add the following line to the Makefile:

```python
python make_md_to_rst.py -p "sphinx-doc/gfoo_md"
```

- The markdown files must be in that subfolders - they cant be gathered in nested subfolders. 

- You can not link to sections in other markdown files, that contain the `-` character.

- The Section structure/ordering must be consistent in the ReST sence (otherwise they wont appear)

- You can link to ressources - like pictures and include them in the markdown, if the pictures are in (possibly another) folder in `\sphinx-doc` and the pathes to this ressources are given relatively!

- You can include a markdown file in a rest document, by appending '_m2r' to the folder name. So to include the markdown file 'foo_md/bar.md' in a toc tree for example - you would do something like:
```python
.. toctree::
   :hidden:
   :maxdepth: 1

   foo_md_m2r/bar
```

- If you want to hyperlink/include other sources from the sphinx documentation that are rest-files, you will not be able to include them in a way, that they will appear in you markdown rendering. - however - there is the slightly hacky possibillity to just include the respective rest directives. This will mess up your markdown code - meaning that you will have those rest snippets flying around, but when the markdown file gets converted to the rest file and build into the sphinx html build, the linked sources will be integrated properly. The syntax is as follows:

- to include the link to the rest source `functions.rst` in the folder `foo`, under the name `bar`, you would need to insert: 
```python
:doc:`foo <rel_path/functions>`
```

- to link to a section with name `foo` in a rest source named `bumm.rst`, under the name `bar`, you would just insert: 
```python
:ref:`bar <relative/path/from/sphinx/root/bumm:foo>`
``` 

- in that manner you might be able to smuggle most rest directives through into the resulting html build. Especially if you want to link to the docstrings of certain (domain specific) objects. Lets say you want to link to the *function* `saqc.funcs.flagRange` under the name `ranger` - you just include:

```python
:py:func:`Ranger <saqc.funcs.flagRange>`
```

whereas the `:func:` part determines the role, the object is documented as. See [this page](https://www.sphinx-doc.org/en/master/#ref-role) for an overview of the available roles

## Refering to documented Functions

* Since the documentation generates an own module structure to document the functions, linking to the documented functions is a bit hacky:

- Functions: to link to any function in the 'saqc.funcs' module - you will have to link the rest file it is documented in. All functions from the function module can be linked via replacing the 'saqc.funcs' part of the module path by 'docs.func_modules':

- For example, 'saqc.funcs.outliers.flagRange' is linked via:
```python
:py:func:`docs.func_modules.outliers.flagRange`
```

To hide the temporal module structure and/or make transparent the intended module structure, use named links, like so:
```python
:py:func:`saqc.outliers.flagRange <docs.func_modules.outliers.flagRange>`
```
