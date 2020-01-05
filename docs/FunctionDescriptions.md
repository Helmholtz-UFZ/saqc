# Implemented Quality Check Functions

Index of the main documentation of the implemented functions, their purpose and parametrization.

## Index

- [Miscellaneous](docs/funcs/Miscellaneous.md)
  - [range](docs/funcs/Miscellaneous.md#range)
  - [seasonalRange](docs/funcs/Miscellaneous.md#seasonalrange)
  - [isolated](docs/funcs/Miscellaneous.md#isolated)
  - [missing](docs/funcs/Miscellaneous.md#missing)
  - [clear](docs/funcs/Miscellaneous.md#clear)
  - [force](docs/funcs/Miscellaneous.md#force)
- [Spike Detection](docs/funcs/SpikeDetection.md)
  - [spikes_basic](docs/funcs/SpikeDetection.md#spikes_basic)
  - [spikes_simpleMad](docs/funcs/SpikeDetection.md#spikes_simplemad)
  - [spikes_slidingZscore](docs/funcs/SpikeDetection.md#spikes_slidingzscore)
  - [spikes_spektrumBased](docs/funcs/SpikeDetection.md#spikes_spektrumbased)
- [Constant Detection](docs/funcs/ConstantDetection.md)
  - [constant](docs/funcs/ConstantDetection.md#constant)
  - [constants_varianceBased](docs/funcs/ConstantDetection.md#constants_variancebased)
- [Break Detection](docs/funcs/BreakDetection.md)
  - [breaks_spektrumBased](docs/funcs/BreakDetection.md#breaks_spektrumbased)
- [Time Series Harmonization](docs/funcs/TimeSeriesHarmonization.md)
  - [harmonize_shift2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_shift2grid)
  - [harmonize_aggregate2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_aggregate2grid)
  - [harmonize_linear2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_linear2grid)
  - [harmonize_interpolate2Grid](docs/funcs/TimeSeriesHarmonization.md#harmonize_interpolate2grid)
  - [harmonize_downsample](docs/funcs/TimeSeriesHarmonization.md#harmonize_downsample)
  - [harmonize](docs/funcs/TimeSeriesHarmonization.md#harmonize)
  - [deharmonize](docs/funcs/TimeSeriesHarmonization.md#deharmonize)
- [Soil Moisture](docs/funcs/SoilMoisture.md)
  - [soilMoisture_spikes](docs/funcs/SoilMoisture.md#soilmoisture_spikes)
  - [soilMoisture_breaks](docs/funcs/SoilMoisture.md#soilmoisture_breaks)
  - [soilMoisture_plateaus](docs/funcs/SoilMoisture.md#soilmoisture_plateaus)
  - [soilMoisture_byFrost](docs/funcs/SoilMoisture.md#soilmoisture_byfrost)
  - [soilMoisture_byPrecipitation](#soilmoisture_byprecipitation)
- [Machine Learning](#machine-learning)
  - [machinelearning](#machinelearning)




## Machine Learning

### machinelearning

```
machinelearning(references, window_values, window_flags, path)
```

| parameter | data type  | default value  | description |
| --------- | ---------- | -------------- | ----------- |
| references    | string or list of strings        |           | the fieldnames of the data series that should be used as reference variables |
| window_values    | integer        |           | Window size that is used to derive the gradients of both the field- and reference-series inside the moving window|
| window_flags   | integer        |          | Window size that is used to count the surrounding automatic flags that have been set before |
| path    | string        |           | Path to the respective model object, i.e. its name and the respective value of the grouping variable. e.g. "models/model_0.2.pkl" |


This Function uses pre-trained machine-learning model objects for flagging. 
This requires training a model by use of the [training script](../ressources/machine_learning/train_machine_learning.py) provided. 
For flagging, inputs to the model are the data of the variable of interest, 
data of reference variables and the automatic flags that were assigned by other 
tests inside SaQC. Therefore, this function should be defined last in the config-file, i.e. it should be the last test that is executed.
Internally, context information for each point is gathered in form of moving 
windows. The size of the moving windows for counting of the surrounding 
automatic flags and for calculation of gradients in the data is specified by 
the user during model training. For the model to work, the parameters 
'references', 'window_values' and 'window_flags' have to be set to the same 
values as during training. For a more detailed description of the modeling 
aproach see the [training script](../ressources/machine_learning/train_machine_learning.py).

