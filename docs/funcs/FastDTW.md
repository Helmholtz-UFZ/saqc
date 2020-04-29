## FastDTW


## Index
[flagFastdtw](#flagFastdtw)

## flagFastDTW

```                            
flagFastDTW(refdatafield='SM1', window = 25, min_distance = 0.25, method_dtw = "fast")
``` 


| parameter             | data type                                                     | default value | description                                                                                                                                                |
|-----------------------|---------------------------------------------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| window                |  int                                                          | `25`          |The number of datapoints to be included in each comparison.                                             |
| min_distance          | float                                                         | `0.5`         |The minimum distance of two graphs to be classified as "different".                                      |
| method_dtw            | string                                                        | `"fast"`      |Implementation of DTW algorithm - "exact" for the normal implementation of DTW, "fast" for the fast implementation.                                                           |
| ref_datafield         | string                                                        |               |Name of the reference datafield ("correct" values) with which the actual datafield is compared.                                             |


This function compares the data with a reference datafield (given in `ref_datafield`) of values we assume to be correct, and flags those values that exceed a minimum distance value (given in `min_distance`). 

As comparison algorithm, we use the [Dynamic Time Warping (DTW) Algorithm](https://en.wikipedia.org/wiki/Dynamic_time_warping) that accounts for temporal and spacial offsets when calculating the distance. For a demonstration of the DTW, see the Wiki entry "Results for rain data set" in [Pattern Recognition with Wavelets](https://git.ufz.de/rdm-software/saqc/-/wikis/Pattern-Recognition-with-Wavelets#Results). 
