Pandas-like indexing
====================

`[]` and `.loc[]`, `.iloc[]` and `.at[]`, `.iat[]` - should behave exactly like 
their counter-parts from pandas.DataFrame. They can take as indexer 
- lists, array-like objects and in general all iterables 
- boolean lists and iterables
- slices
- scalars and any hashable object

Most indexers are directly passed to the underling columns-series or row-series depending 
on the position of the indexer and the complexity of the operation. For `.loc`, `.iloc`, `.at` 
and `iat` the first position is the *row indexer*, the second the *column indexer*. The second 
can be omitted and will default to `slice(None)`. Examples:
- `di.loc[[1,2,3], ['a']]` : select labels 1,2,3 from column a
- `di.iloc[[1,2,3], [0,3]]` : select positions 1,2,3 from the columns 0 and 3
- `di.loc[:, 'a':'c']` : select all rows from columns a to d 
- `di.at[4,'c']` : select the elements with label 4 in column c
- `di.loc[:]` -> `di.loc[:,:]` : select everything. 

Scalar indexing always return a pandas Series if the other indexer is a non-scalar. If both indexer
are scalars, the element itself is returned. In all other cases a dios is returned. 
For more pandas-like indexing magic and the differences between the indexers, 
see the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html).

>**Note:**
>
>In contrast to pandas.DataFrame, `.loc[:]` and `.loc[:, :]` always behaves identical. Same apply for `iloc` and
>[`aloc`](#the-special-indexer-aloc). For example, two pandas.DataFrames `df1` and `df2` with different columns,
>does align columns with `df1.loc[:, :] = df2` , but does **not** with `df1.loc[:] = df2`. 
>
>If this is the desired behavior or a bug, i couldn't verify so far. -- Bert Palm 

**2D-indexer**

`dios[boolean dios-like]` (as single key) -  dios accept boolean 2D-indexer (boolean pandas.Dataframe 
or boolean Dios). 

Columns and rows from the indexer align with the dios. 
This means that only matching columns selected and in this columns rows are selected where 
i) indices are match and ii) the value is True in the indexer-bool-dios. There is no difference between
missing indices and present indices, but False values.

Values from unselected rows and columns are dropped, but empty columns are still preserved, 
with the effect that the resulting Dios always have the same column dimension than the initial dios. 

>**Note:**
>This is the exact same behavior like pandas.DataFrame's handling of 2D-indexer, despite that pandas.DataFrame 
>fill numpy.nan's at missing locations and therefore also fill-up, whole missing columns with numpy.nan's.

**setting values**

Setting values with `[]` and `.loc[]`, `.iloc[]` and `.at[]`, `.iat[]` works like in pandas. 
With `.at`/`.iat` only single items can be set, for the other the
right hand side values can be:
 - *scalars*: these are broadcasted to the selected positions
 - *lists*: the length the list must match the number of indexed columns. The items can be everything that 
    can applied to a series, with the respective indexing method (`loc`, `iloc`, `[]`).
 - *dios*: the length of the columns must match the number of indexed columns - columns does *not* align, 
    they are just iterated. 
    Rows do align. Rows that are present on the right but not on the left are ignored. 
    Rows that are present on the left (bear in mind: these rows was explicitly chosen for write!), but not present
    on the right, are filled with `NaN`s, like in pandas.
 - *pandas.Series*: column indexer must be a scalar(!), the series is passed down, and set with `loc`, `iloc` or `[]` 
    by pandas Series, where it maybe align, depending on the method. 

**Examples:**

- `dios.loc[2:5, 'a'] = [1,2,3]` is the same as `a=dios['a']; a.loc[2:5]=[1,2,3]; dios['a']=a`
- `dios.loc[2:5, :] = 99` : set 99 on rows 2 to 5 on all columns

Special indexer `.aloc`
========================

Additional to the pandas like indexers we have a `.aloc[..]` (align locator) indexing method. 
Unlike `.iloc` and `.loc` indexers fully align if possible and 1D-array-likes can be broadcast 
to multiple columns at once. This method also handle missing indexer-items gracefully. 
It is used like `.loc`, so a single indexer (`.aloc[indexer]`) or a tuple of row-indexer and 
column-indexer (`.aloc[row-indexer, column-indexer]`) can be given. Also it can handle boolean and *non-bolean*
2D-Indexer.

The main **purpose** of `.aloc` is:
- to select gracefully, so rows or columns, that was given as indexer, but doesn't exist, not raise an error
- align series/dios-indexer 
- vertically broadcasting aka. setting multiple columns at once with a list-like value

Aloc usage
----------

`aloc` is *called* like `loc`, with a single key, that act as row indexer `aloc[rowkey]` or with a tuple of
row indexer and column indexer `aloc[rowkey, columnkey]`. Also 2D-indexer (like dios or df) can be given, but 
only as a single key, like `.aloc[2D-indexer]` or with the special column key `...`, 
the ellipsis (`.aloc[2D-indexer, ...]`). The ellipsis may change, how the 2D-indexer is
interpreted, but this will explained [later](#the-power-of-2d-indexer) in detail.

If a normal (non 2D-dimensional) row indexer is given, but no column indexer, the latter defaults to `:` aka. 
`slice(None)`, so `.aloc[row-indexer]` becomes `.aloc[row-indexer, :]`, which means, that all columns are used.
In general, a normal row-indexer is applied to every column, that was chosen by the column indexer, but for 
each column separately.

So maybe a first example gives an rough idea:
```
>>> s = pd.Series([11] * 4 )
>>> di = DictOfSeries(dict(a=s[:2]*6, b=s[2:4]*7, c=s[:2]*8, d=s[1:3]*9))
>>> di
    a |     b |     c |     d | 
===== | ===== | ===== | ===== | 
0  66 | 2  77 | 0  88 | 1  99 | 
1  66 | 3  77 | 1  88 | 2  99 | 


>>> di.aloc[[1,2], ['a', 'b', 'd', 'x']]
    a |     b |     d | 
===== | ===== | ===== | 
1  66 | 2  77 | 1  99 | 
      |       | 2  99 | 
```

The return type
----------------

Unlike the other two indexer methods `loc` and `iloc`, it is not possible to get a single item returned; 
the return type is either a pandas.Series, iff the column-indexer is a single key (eg. `'a'`) or a dios, iff not.
The row-indexer does not play any role in the return type choice.

> **Note for the curios:** 
> 
> This is because a scalar (`.aloc[key]`) is translates to `.loc[key:key]` under the hood.

Indexer types
-------------
Following the `.aloc` specific indexer are listed. Any indexer that is not listed below (slice, boolean lists, ...), 
but are known to work with `.loc`, are treated as they would passed to `.loc`, as they actually do under the hood.

Some indexer are linked to later sections, where a more detailed explanation and examples are given.

*special [Column indexer](#select-columns-gracefully) are :*
- *list / array-like* (or any iterable object): Only labels that are present in the columns are used, others are 
   ignored. 
- *pd.Series* : `.values` are taken from series and handled like a *list*.
- *scalar* (or any hashable obj) : Select a single column, if label is present, otherwise nothing. 


*special [Row indexer](#selecting-rows-a-smart-way) are :*
- *list / array-like* (or any iterable object): Only rows, which indices are present in the index of the column are 
   used, others are ignored. A dios is returned. 
- *scalar* (or any hashable obj) : Select a single row from a column, if the value is present in the index of 
   the column, otherwise nothing is selected. [1]
- *pd.Series* : align the index from the given Series with the column, what means only common indices are used. The 
   actual values of the series are ignored(!).
- *boolean pd.Series* : like *pd.Series* but only True values are evaluated. 
   False values are equivalent to missing indices. To treat a boolean series as a *normal* indexer series, as decribed
   above, one can use `.aloc(usebool=False)[boolean pd.Series]`.
   

*special [2D-indexer](#the-power-of-2d-indexer) are :*
- `.aloc[boolean dios-like]` : work same like `di[boolean dios-like]` (see there). 
   Brief: full align, select items, where the index is present and the value is True.
- `.aloc[dios-like, ...]` (with Ellipsis) : Align in columns and rows, ignore its values. Per common column,
   the common indices are selected. The ellipsis forces `aloc`, to ignore the values, so a boolean dios could be 
   treated as a non-boolean. Alternatively `.aloc(usebool=False)[boolean dios-like]` could be used.[2]
- `.aloc[nested list-like]` : The inner lists are used as `aloc`-*list*-row-indexer (see there) on all columns. 
   One list for one column, which implies, that the outer list has the same length as the number of columns. 

*special handling of 1D-**values***

Values that are list- or array-like, which includes pd.Series, are set on all selected columns. pd.Series align
like `s1.loc[:] = s2` do. See also the [cookbook](/docs/cookbook.md#broadcast-array-likes-to-multiple-columns).


Aloc overiew table
---------------------

| example | type | on  | like `.loc` | handling | conditions / hints | link |
| ------- | ---- | --- | ----------- | -------- | ------------------ | ---- |
| `.aloc[any, 'a']`         | scalar               | columns |no | select graceful | - | [cols](#select-columns-gracefully)|
|[Column indexer](#select-columns-gracefully)| 
| `.aloc[any, 'a']`         | scalar               | columns |no | select graceful | - | [cols](#select-columns-gracefully)|
| `.aloc[any, 'b':'z']`       | slice                | columns |yes| slice | - | [cols](#select-columns-gracefully)|
| `.aloc[any, ['a','c']]`     | list-like            | columns |no | filter graceful | - | [cols](#select-columns-gracefully)|
| `.aloc[any [True,False]]`   | bool list-like       | columns |yes| take `True`'s | length must match nr of columns | [cols](#select-columns-gracefully)|
| `.aloc[any, s]`             | Series        | columns |no | like list,  | only `s.values` are evaluated | [cols](#select-columns-gracefully)|
| `.aloc[any, bs]`            | bool Series   | columns |yes| like bool-list | see there | [cols](#select-columns-gracefully)|
|[Row indexer](#selecting-rows-a-smart-way)|  
| `.aloc[7, any]`             | scalar               | rows    |no | translate to `.loc[key:key]` | - | [rows](#selecting-rows-a-smart-way) |
| `.aloc[3:42, any]`          | slice                | rows    |yes| slice | - | | 
| `.aloc[[1,2,24], any]`      | list-like            | rows    |no | filter graceful | - | [rows](#selecting-rows-a-smart-way) |
| `.aloc[[True,False], any]`  | bool list-like       | rows    |yes| take `True`'s | length must match nr of (all selected) columns | [blist](#boolean-array-likes-as-row-indexer)|
| `.aloc[s, any]`             | Series        | rows    |no | like `.loc[s.index]` | - | [ser](#pandasseries-and-boolean-pandasseries-as-row-indexer) |
| `.aloc[bs, any]`            | bool Series   | rows    |no | align + just take `True`'s  | evaluate `usebool`-keyword |  [ser](#pandasseries-and-boolean-pandasseries-as-row-indexer)|
| `.aloc[[[s],[1,2,3]], any]` | nested list-like     | both    | ? | one row-indexer per column | outer length must match nr of (selected) columns | [nlist](#nested-lists-as-row-indexer) |
|[2D-indexer](#the-power-of-2d-indexer)| 
| `.aloc[di]`                 | dios-like            | both    |no | full align  | - | |
| `.aloc[di, ...]`            | dios-like            | both    |no | full align | ellipsis has no effect | |
| `.aloc[di>5]`               | bool dios-like       | both    |no | full align + take `True`'s | evaluate `usebool`-keyword | |
| `.aloc[di>5, ...]`          | (bool) dios-like     | both    |no | full align, **no** bool evaluation | - | |

Example dios
------------

The Dios used in the examples, unless stated otherwise, looks like so:

```
>>> dictofser
    a |      b |      c |     d | 
===== | ====== | ====== | ===== | 
0   0 | 2    5 | 4    7 | 6   0 | 
1   7 | 3    6 | 5   17 | 7   1 | 
2  14 | 4    7 | 6   27 | 8   2 | 
3  21 | 5    8 | 7   37 | 9   3 | 
4  28 | 6    9 | 8   47 | 10  4 | 
5  35 | 7   10 | 9   57 | 11  5 | 
6  42 | 8   11 | 10  67 | 12  6 | 
7  49 | 9   12 | 11  77 | 13  7 | 
8  56 | 10  13 | 12  87 | 14  8 | 
```

or the short version:

```
>>> di
    a |    b |     c |     d | 
===== | ==== | ===== | ===== | 
0   0 | 2  5 | 4   7 | 6   0 | 
1   7 | 3  6 | 5  17 | 7   1 | 
2  14 | 4  7 | 6  27 | 8   2 | 
3  21 | 5  8 | 7  37 | 9   3 | 
4  28 | 6  9 | 8  47 | 10  4 | 
```

The example Dios can get via a function:

```
from dios import example_DictOfSeries()
mydios = example_DictOfSeries()
```

or generated manually like so:

``` 
>>> a = pd.Series(range(0, 70, 7))
>>> b = pd.Series(range(5, 15, 1))
>>> c = pd.Series(range(7, 107, 10))
>>> d = pd.Series(range(0, 10, 1))
>>> for i, s in enumerate([a,b,c,d]): s.index += i*2
>>> dictofser = DictOfSeries(dict(a=a, b=b, c=c, d=d))
>>> di = dictofser[:5]
```


Select columns, gracefully
---------------------------

One can use `.aloc[:, key]` to select **single columns** gracefully. 
The underling pandas.Series is returned, if the key exist. 
Otherwise a empty pandas.Series with `dtype=object` is returned.

```
>>> di.aloc[:, 'a']
0     0
1     7
2    14
3    21
4    28
Name: a, dtype: int64

>>> di.aloc[:, 'x']
Series([], dtype: object)
```


**Multiple columns**

Just like selecting *single columns gracefully*, but with a array-like indexer. 
A dios is returned, with a subset of the existing columns. 
If no key is present a empty dios is returned. 

```
>>> di.aloc[:, ['c', 99, None, 'a', 'x', 'y']]
    a |     c | 
===== | ===== | 
0   0 | 4   7 | 
1   7 | 5  17 | 
2  14 | 6  27 | 
3  21 | 7  37 | 
4  28 | 8  47 | 

>>> di.aloc[:, ['x', 'y']]
Empty DictOfSeries
Columns: []

s = pd.Series(dict(a='a', b='x', c='c', foo='d'))
d.aloc[:, s]
    a |     c |     d | 
===== | ===== | ===== | 
0   0 | 4   7 | 6   0 | 
1   7 | 5  17 | 7   1 | 
2  14 | 6  27 | 8   2 | 
3  21 | 7  37 | 9   3 | 
4  28 | 8  47 | 10  4 | 
```

**Boolean indexing, indexing with pd.Series and slice indexer**

**Boolean indexer**, for example `[True, 'False', 'True', 'False']`, must have the same length than the number 
of columns, then only columns, where the indexer has a `True` value are selected.

If the key is a **pandas.Series**, its *values* are used for indexing, especially the Series's index is ignored. If a 
series has boolean values its treated like a boolean indexer, otherwise its treated as a array-like indexer.

A easy way to select all columns, is, to use null-**slice**es, like `.aloc[:,:]` or even simpler `.aloc[:]`. 
This is just like one would do, with `loc` or `iloc`. Of course slicing with boundaries also work, 
eg `.loc[:, 'a':'f']`. 

>**See also**
> - [pandas slicing ranges](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#slicing-ranges) 
> - [pandas boolean indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing)


Selecting Rows a smart way
--------------------------

For scalar and array-like indexer with label values, the keys are handled gracefully, just like with 
array-like column indexers.

``` 
>>> di.aloc[1]
   a |       b |       c |       d | 
==== | ======= | ======= | ======= | 
1  7 | no data | no data | no data | 

>>> di.aloc[99]
Empty DictOfSeries
Columns: ['a', 'b', 'c', 'd']

>>> di.aloc[[3,6,7,18]]
    a |    b |     c |    d | 
===== | ==== | ===== | ==== | 
3  21 | 3  6 | 6  27 | 6  0 | 
      | 6  9 | 7  37 | 7  1 | 
```

The length of columns can differ:
``` 
>>> di.aloc[[3,6,7,18]].aloc[[3,6]]
    a |    b |     c |    d | 
===== | ==== | ===== | ==== | 
3  21 | 3  6 | 6  27 | 6  0 | 
      | 6  9 |       |      | 
```

Boolean array-likes as row indexer
---------------------------------

For array-like indexer that hold boolean values, the length of the indexer and
the length of all column(s) to index must match.
``` 
>>> di.aloc[[True,False,False,True,False]]
    a |    b |     c |    d | 
===== | ==== | ===== | ==== | 
0   0 | 2  5 | 4   7 | 6  0 | 
3  21 | 5  8 | 7  37 | 9  3 | 
```
If the length does not match a `IndexError` is raised:
```
>>> di.aloc[[True,False,False]]
Traceback (most recent call last):
  ...
  IndexError: failed for column a: Boolean index has wrong length: 3 instead of 5
```

This can be tricky, especially if columns have different length:
``` 
>>> difflen
    a |    b |     c |    d | 
===== | ==== | ===== | ==== | 
0   0 | 2  5 | 4   7 | 6  0 | 
1   7 | 3  6 | 6  27 | 7  1 | 
2  14 | 4  7 |       | 8  2 | 

>>> difflen.aloc[[False,True,False]]
Traceback (most recent call last):
  ...
  IndexError: Boolean index has wrong length: 3 instead of 2
```

pandas.Series and boolean pandas.Series as row indexer
------------------------------------------------------

When using a pandas.Series as row indexer with `aloc`, all its magic comes to light.
The index of the given series align itself with the index of each column separately and is this way used as a filter.

```
>>> s = di['b'] + 100
>>> s
2    105
3    106
4    107
5    108
6    109
Name: b, dtype: int64

>>> di.aloc[s]
    a |    b |     c |    d | 
===== | ==== | ===== | ==== | 
2  14 | 2  5 | 4   7 | 6  0 | 
3  21 | 3  6 | 5  17 |      | 
4  28 | 4  7 | 6  27 |      | 
      | 5  8 |       |      | 
      | 6  9 |       |      | 
```

As seen in the example above the series' values are ignored completely. The functionality  
is similar to `s1.loc[s2.index]`, with `s1` and `s2` are pandas.Series's, and s2 is the indexer and s1 is one column 
after the other.

If the indexer series holds boolean values, these are **not** ignored. 
The series align the same way as explained above, but additional only the `True` values are evaluated. 
Thus `False`-values are treated like missing indices. The behavior here is analogous to `s1.loc[s2[s2].index]`.

``` 
>>> boolseries = di['b'] > 6
>>> boolseries
2    False
3    False
4     True
5     True
6     True
Name: b, dtype: bool

>>> di.aloc[boolseries]
    a |    b |     c |    d | 
===== | ==== | ===== | ==== | 
4  28 | 4  7 | 4   7 | 6  0 | 
      | 5  8 | 5  17 |      | 
      | 6  9 | 6  27 |      | 
```

To evaluate boolean values is a very handy feature, as it can easily used with multiple conditions and also fits
nicely with writing those as one-liner:

``` 
>>> di.aloc[d['b'] > 6]
    a |    b |     c |    d | 
===== | ==== | ===== | ==== | 
4  28 | 4  7 | 4   7 | 6  0 | 
      | 5  8 | 5  17 |      | 
      | 6  9 | 6  27 |      | 

>>> di.aloc[(d['a'] > 6) & (d['b'] > 6)]
    a |    b |    c |       d | 
===== | ==== | ==== | ======= | 
4  28 | 4  7 | 4  7 | no data | 
```


>**Note:**
>
>Nevertheless, something like `di.aloc[di['a'] > di['b']]` do not work, because the comparison fails, 
>as long as the two series objects not have the same index. But maybe one want to checkout 
>[DictOfSeries.index_of()](https://dios.readthedocs.io/en/latest/_api/dios.DictOfSeries.html#dios.DictOfSeries.index_of).


Nested-lists as row indexer
---------------------------

It is possible to pass different array-like indexer to different columns, by using nested lists as indexer. 
The outer list's length must match the number of columns of the dios. The items of the outer list, all must be
array-like and not further nested. For example list, pandas.Series, boolean lists or pandas.Series, numpy.arrays...
Every inner list-like item is applied as row indexer to the according column. 

``` 
>>> d
    a |    b |     c |     d | 
===== | ==== | ===== | ===== | 
0   0 | 2  5 | 4   7 | 6   0 | 
1   7 | 3  6 | 5  17 | 7   1 | 
2  14 | 4  7 | 6  27 | 8   2 | 
3  21 | 5  8 | 7  37 | 9   3 | 
4  28 | 6  9 | 8  47 | 10  4 | 

>>> di.aloc[ [d['a'], [True,False,True,False,False], [], [7,8,10]] ]
    a |    b |       c |     d | 
===== | ==== | ======= | ===== | 
0   0 | 2  5 | no data | 7   1 | 
1   7 | 4  7 |         | 8   2 | 
2  14 |      |         | 10  4 | 
3  21 |      |         |       | 
4  28 |      |         |       | 

>>> ar = np.array([2,3])
>>> di.aloc[[ar, ar+1, ar+2, ar+3]]
    a |    b |     c |    d | 
===== | ==== | ===== | ==== | 
2  14 | 3  6 | 4   7 | 6  0 | 
3  21 | 4  7 | 5  17 |      | 
```

Even this looks like a 2D-indexer, that are explained in the next section, it is not. 
In contrast to the 2D-indexer, we also can provide a column key, to pre-filter the columns.

```
>>> di.aloc[[ar, ar+1, ar+3], ['a','b','d']]
    a |    b |    d | 
===== | ==== | ==== | 
2  14 | 3  6 | 6  0 | 
3  21 | 4  7 |      | 
```



The power of 2D-indexer
-----------------------

Overview: 

|                      |        |
| ------               | ------ |
| `.aloc[bool-dios]`         | 1. align columns, 2. align rows, 3. just take `True`'s  -- [1] |
| `.aloc[dios, ...]` (use Ellipsis)        | 1. align columns, 2. align rows, (3.) ignore values  -- [1] |
[1] evaluate `usebool`-keyword


**T_O_D_O**

