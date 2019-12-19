## Offset Strings
All the [pandas offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) ars supported by SaQC. The following table lists some of the more relevant options:

| Alias          | Description  |
| -----          | -----------  |
| `"S"`          | second       |
| `"T"`, `"Min"` | minute       |
| `"H"`          | hour         |
| `"D"`          | calendar day |
| `"W"`          | week         |
| `"M"`          | month        |
| `"Y"`          | year         |

Multiples are build by preceeding the alias with the desired multiply (e.g `"5Min"`, `"4W"`)


## Constants

### Flagging Constants
The following flag constants are available and can be used to mark the quality of a data point:

| Alias       | Description                                                                                   |
| ----        | ----                                                                                          |
| `GOOD`      | A value did pass all the test and is therefore considered to be valid                         |
| `BAD`       | At least on test failed on the values and is therefore considered to be invalid               |
| `UNFLAGGED` | The value has not got a flag yet. This might mean, that all tests passed or that no tests ran |

How these aliases will be translated into 'real' flags (output of SaQC) dependes on the flagger implementation
and might range from numerical values to string concstants

### Numerical Constants
| Alias    | Description  |
| ----     | ----         |
| `NAN`    | Not a number |
| `NODATA` | Missing data |
