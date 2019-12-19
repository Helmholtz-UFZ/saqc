## Offset Strings
All the [pandas offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) ars supported by SaQC. The following table lists some of the more relevant options:
| Alias          | Description  |
| -----          | -----------  |
| `"S"`          | second       |
| `"T"`, `"Min"` | minute       |
| `H`            | hour         |
| `D`            | calendar day |
| `W`            | week         |
| `M`            | month        |
| `Y`            | year         |
Multiples are build by preceeding the alias with the desired multiply (e.g `"5Min"`, `"4W"`)

