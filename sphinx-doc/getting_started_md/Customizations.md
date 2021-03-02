# Customizations
SaQC comes with a continuously growing number of pre-implemented
[quality check and processing routines](sphinx-doc/getting_started_md/FunctionIndex.md) and 
flagging schemes. 
For any sufficiently large use case however it is very likely that the 
functions provided won't fulfill all your needs and requirements.

Acknowledging the impossibility to address all imaginable use cases, we 
designed the system to allow for extensions and costumizations. The main extensions options, namely 
[quality check routines](#custom-quality-check-routines)
and the [flagging scheme](#custom-flagging-schemes)
are described within this documents.

## Custom quality check routines
In case you are missing quality check routines, you are of course very
welcome to file a feature request issue on the project's
[gitlab repository](https://git.ufz.de/rdm-software/saqc). However, if 
you are more the "no-way-I-get-this-done-by-myself" type of person,
SaQC provides two ways to integrate custom routines into the system:
1. The [extension language](sphinx-doc/getting_started_md/GenericFunctions.md)
2. An [interface](#interface) to the evaluation machinery

### Interface
In order to make a function usable within the evaluation framework of SaQC the following interface is needed:

```python
def yourTestFunction(
   data: pandas.DataFrame,
   field: str,
   flagger: saqc.flagger.BaseFlagger,
   *args: Any,
   **kwargs: Any
   ) -> (dios.DictOfSeries, saqc.flagger.BaseFlagger)
```

#### Argument Descriptions

| Name      | Description                                                                                      |
|-----------|--------------------------------------------------------------------------------------------------|
| `data`    | The actual dataset.                                                                               |
| `field`   | The field/column within `data`, that function is processing.                              |
| `flagger` | An instance of a flagger, responsible for the translation of test results into quality attributes. |
| `args`    | Any other arguments needed to parameterize the function.                                          |
| `kwargs`  | Any other keyword arguments needed to parameterize the function.                                  |

### Integrate into SaQC
In order make your function available to the system it needs to be registered. We provide the decorator 
[`register`](saqc/functions/register.py) in the module `saqc.functions.register` to integrate your 
test functions into SaQC. Here is a complete dummy example:

```python
from saqc.functions.register import register

@register
def yourTestFunction(data, field, flagger, *args, **kwargs):
    return data, flagger
```

### Example
The function [`flagRange`](saqc/funcs/functions.py) provides a simple, yet complete implementation of 
a quality check routine. You might want to look into its implementation as a reference for your own.


## Custom flagging schemes
Sorry for the inconvenience! Coming soon...
