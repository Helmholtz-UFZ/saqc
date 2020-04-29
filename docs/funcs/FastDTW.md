## Flag FastDTW


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
| ref_datafield          | string                                                       |               |Name of the reference datafield with which the actual datafield is compared.                                             |


