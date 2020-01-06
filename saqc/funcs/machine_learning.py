import pandas as pd
import numpy as np
from saqc.funcs.register import register

# NEW
from sklearn.ensemble import RandomForestClassifier
import joblib


def _refCalc(reference, window_values):
    # Helper function for calculation of moving window values
    outdata = pd.DataFrame()
    name = reference.name
    # derive gradients from reference series
    outdata[name + "_Dt_1"] = reference - reference.shift(1)  # gradient t vs. t-1
    outdata[name + "_Dt1"] = reference - reference.shift(-1)  # gradient t vs. t+1
    # moving mean of gradients var1 and var2 before/after
    outdata[name + "_Dt_" + str(window_values)] = (
        outdata[name + "_Dt_1"].rolling(window_values, center=False).mean()
    )  # mean gradient t to t-window
    outdata[name + "_Dt" + str(window_values)] = (
        outdata[name + "_Dt_1"].iloc[::-1].rolling(window_values, center=False).mean()[::-1]
    )  # mean gradient t to t+window
    return outdata


@register("machinelearning")
def flagML(data, field, flagger, references, window_values: int, window_flags: int, path: str, **kwargs):

    """This Function uses pre-trained machine-learning model objects for flagging of a specific variable. The model is supposed to be trained using the script provided in "ressources/machine_learning/train_machine_learning.py".
    For flagging, Inputs to the model are the timeseries of the respective target at one specific sensors, the automatic flags that were assigned by SaQC as well as multiple reference series.
    Internally, context information for each point is gathered in form of moving windows to improve the flagging algorithm according to user input during model training.
    For the model to work, the parameters 'references', 'window_values' and 'window_flags' have to be set to the same values as during training.
    :param data:                        The pandas dataframe holding the data-to-be flagged, as well as the reference series. Data must be indexed by a datetime index.
    :param flags:                       A dataframe holding the flags
    :param field:                       Fieldname of the field in data that is to be flagged.
    :param flagger:                     A flagger - object.
    :param references:                  A string or list of strings, denoting the fieldnames of the data series that should be used as reference variables
    :param window_values:               An integer, denoting the window size that is used to derive the gradients of both the field- and reference-series inside the moving window
    :param window_flags:                An integer, denoting the window size that is used to count the surrounding automatic flags that have been set before
    :param path:                        A string giving the path to the respective model object, i.e. its name and the respective value of the grouping variable. e.g. "models/model_0.2.pkl"
    """

    # Function for moving window calculations
    # Create custom df for easier processing
    df = data.loc[:, [field] + references]
    # Create binary column of BAD-Flags
    df["flag_bin"] = flagger.isFlagged(field, flag=flagger.BAD, comparator="==").astype(
        "int"
    )  # get "BAD"-flags and turn into binary

    # Add context information of flags
    df["flag_bin_t_1"] = df["flag_bin"] - df["flag_bin"].shift(1)  # Flag at t-1
    df["flag_bin_t1"] = df["flag_bin"] - df["flag_bin"].shift(-1)  # Flag at t+1
    df["flag_bin_t_" + str(window_flags)] = (
        df["flag_bin"].rolling(window_flags + 1, center=False).sum()
    )  # n Flags in interval t to t-window_flags
    df["flag_bin_t" + str(window_flags)] = (
        df["flag_bin"].iloc[::-1].rolling(window_flags + 1, center=False).sum()[::-1]
    )  # n Flags in interval t to t+window_flags
    # forward-orientation not possible, so right-orientation on reversed data an reverse result

    # Add context information for field+references
    for i in [field] + references:
        df = pd.concat([df, _refCalc(reference=df[i], window_values=window_values)], axis=1)

    # remove rows that contain NAs (new ones occured during predictor calculation)
    df = df.dropna(axis=0, how="any")
    # drop column of automatic flags at time t
    df = df.drop(columns="flag_bin")
    # Load model and predict on df:
    model = joblib.load(path)
    preds = model.predict(df)

    # Get indices of flagged values
    flag_indices = df[preds.astype("bool")].index
    # set Flags
    flagger = flagger.setFlags(field, loc=flag_indices, **kwargs)
    return data, flagger
