Itype
=====

DictOfSeries holds multiple series, and each series can have a different index length 
and index type. Differing index lengths are either solved by some aligning magic, or simply fail, if 
aligning makes no sense (eg. assigning the very same list to series of different lengths (see `.aloc`).

A bigger challange is the type of the index. If one series has an alphabetical index, and another one 
a numeric index, selecting along columns can fail in every scenario. To keep track of the
types of index or to prohibit the inserting of a *not fitting* index type, 
we introduce the `itype`. This can be set on creation of a Dios and also changed during usage. 
On change of the itype, all indexes of all series in the dios are casted to a new fitting type,
if possible. Different cast-mechanisms are available. 

If an itype prohibits some certain types of indexes and a series with a non-fitting index-type is inserted, 
an implicit type cast is done (with or without a warning) or an error is raised. The warning/error policy
can be adjusted via global options. 

