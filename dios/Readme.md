DictOfSeries 
============

DictOfSeries is a pandas.Series of pandas.Series objects which aims to behave as similar as possible to pandas.DataFrame. 


Nomenclature
------------
- series/ser: instance of pandas.Series
- dios: instance of dios.DictOfSeries
- df: instance of pandas.DataFrame
- dios-like: a *dios* or a *df*
- alignable object: a *dios*, *df* or a *series*


Features
--------
* every *column* has its own index
* uses much less memory than a misaligned pandas.DataFrame
* behaves quite like a pandas.DataFrame
* additional align locator (`.aloc[]`)

Install
-------

todo: PyPi

``` 
import dios

# Have fun :)
```

Documentation
-------------

The main docu is on ReadTheDocs at: 

* [dios.rtfd.io](https://dios.rtfd.io)

but some docs are also available local:
* [Indexing](/docs/doc_indexing.md)
* [Cookbook](/docs/doc_cookbook.md)
* [Itype](/docs/doc_itype.md)

TL;DR
-----
**get it**
```
>>> from dios import DictOfSeries
```
**empty**
```
>>> DictOfSeries()
Empty DictOfSeries
Columns: []

>>> DictOfSeries(columns=['x', 'y'])
Empty DictOfSeries
Columns: ['x', 'y']

>>> DictOfSeries(columns=['x', 'y'], index=[3,4,5])
     x |      y | 
====== | ====== | 
3  NaN | 3  NaN | 
4  NaN | 4  NaN | 
5  NaN | 5  NaN | 
```
**with data**
```
>>> DictOfSeries([range(4), range(2), range(3)])
   0 |    1 |    2 | 
==== | ==== | ==== | 
0  0 | 0  0 | 0  0 | 
1  1 | 1  1 | 1  1 | 
2  2 |      | 2  2 | 
3  3 |      |      | 

>>> DictOfSeries(np.random.random([2,4]))
          0 |           1 | 
=========== | =========== | 
0  0.112020 | 0  0.509881 | 
1  0.108070 | 1  0.285779 | 
2  0.851453 | 2  0.805933 | 
3  0.138352 | 3  0.812339 | 

>>> DictOfSeries(np.random.random([2,4]), columns=['a','b'], index=[11,12,13,14])
           a |            b | 
============ | ============ | 
11  0.394304 | 11  0.356206 | 
12  0.943689 | 12  0.735356 | 
13  0.791820 | 13  0.066947 | 
14  0.759802 | 14  0.496321 | 

>>> DictOfSeries(dict(today=['spam']*3, tomorrow=['spam']*2))
  today |   tomorrow | 
======= | ========== | 
0  spam | 0     spam | 
1  spam | 1     spam | 
2  spam |            | 
```

