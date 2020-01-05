# Machine Learning

A collection of interfaces to pre-trained machine learning models.

## Index
- [machinelearning](#machinelearning)


## machinelearning

```
machinelearning(references, window_values, window_flags, path)
```

| parameter     | data type                 | default value | description                                                                                                                       |
|---------------|---------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------|
| references    | string or list of strings |               | the field names of the data series that should be used as reference variables                                                      |
| window_values | integer                   |               | Window size that is used to derive the gradients of both the field- and reference-series inside the moving window                 |
| window_flags  | integer                   |               | Window size that is used to count the surrounding automatic flags that have been set before                                       |
| path          | string                    |               | Path to the respective model object, i.e. its name and the respective value of the grouping variable. e.g. "models/model_0.2.pkl" |


This Function uses pre-trained machine-learning model objects for flagging. 
This requires training a model by use of the [training script](../ressources/machine_learning/train_machine_learning.py) provided. 
For flagging, inputs to the model are the data of the variable of interest, 
data of reference variables and the automatic flags that were assigned by other 
tests inside SaQC. Therefore, this function should be executed after all other tests.
Internally, context information for each point is gathered in form of moving 
windows. The size of the moving windows for counting of the surrounding 
automatic flags and for calculation of gradients in the data is specified by 
the user during model training. For the model to work, the parameters 
'references', 'window_values' and 'window_flags' have to be set to the same 
values as during training. For a more detailed description of the modeling 
approach see the [training script](ressources/machine_learning/train_machine_learning.py).

