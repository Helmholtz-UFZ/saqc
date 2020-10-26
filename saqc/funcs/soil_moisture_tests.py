#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import joblib
import dios
from scipy.signal import savgol_filter

from saqc.funcs.breaks_detection import breaks_flagSpektrumBased
from saqc.funcs.spikes_detection import spikes_flagSpektrumBased
from saqc.funcs.constants_detection import constants_flagVarianceBased
from saqc.core.register import register
from saqc.lib.tools import retrieveTrustworthyOriginal


@register(masking='field')
def sm_flagSpikes(
    data,
    field,
    flagger,
    raise_factor=0.15,
    deriv_factor=0.2,
    noise_func="CoVar",
    noise_window="12h",
    noise_thresh=1,
    smooth_window="3h",
    smooth_poly_deg=2,
    **kwargs,
):

    """
    The Function provides just a call to ``flagSpikes_spektrumBased``, with parameter defaults,
    that refer to References [1].

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    raise_factor : float, default 0.15
        Minimum relative value difference between two values to consider the latter as a spike candidate.
        See condition (1) (or reference [2]).
    deriv_factor : float, default 0.2
        See condition (2) (or reference [2]).
    noise_func : {'CoVar', 'rVar'}, default 'CoVar'
        Function to calculate noisiness of the data surrounding potential spikes.
        ``'CoVar'``: Coefficient of Variation
        ``'rVar'``: Relative Variance
    noise_window : str, default '12h'
        An offset string that determines the range of the time window of the "surrounding" data of a potential spike.
        See condition (3) (or reference [2]).
    noise_thresh : float, default 1
        Upper threshold for noisiness of data surrounding potential spikes. See condition (3) (or reference [2]).
    smooth_window : {None, str}, default None
        Size of the smoothing window of the Savitsky-Golay filter.
        The default value ``None`` results in a window of two times the sampling rate (i.e. containing three values).
    smooth_poly_deg : int, default 2
        Degree of the polynomial used for fitting with the Savitsky-Golay filter.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional Informations related to `data`.
        Flags values may have changed relatively to the flagger input.

    References
    ----------
    This Function is a generalization of the Spectrum based Spike flagging mechanism as presented in:

    [1] Dorigo, W. et al: Global Automated Quality Control of In Situ Soil Moisture
        Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
        doi:10.2136/vzj2012.0097.

    [2] https://git.ufz.de/rdm-software/saqc/-/blob/testfuncDocs/docs/funcs/FormalDescriptions.md#spikes_flagspektrumbased

    """

    return spikes_flagSpektrumBased(
        data,
        field,
        flagger,
        raise_factor=raise_factor,
        deriv_factor=deriv_factor,
        noise_func=noise_func,
        noise_window=noise_window,
        noise_thresh=noise_thresh,
        smooth_window=smooth_window,
        smooth_poly_deg=smooth_poly_deg,
        **kwargs,
    )


@register(masking='field')
def sm_flagBreaks(
    data,
    field,
    flagger,
    thresh_rel=0.1,
    thresh_abs=0.01,
    first_der_factor=10,
    first_der_window="12h",
    scnd_der_ratio_range=0.05,
    scnd_der_ratio_thresh=10,
    smooth=False,
    smooth_window="3h",
    smooth_poly_deg=2,
    **kwargs,
):

    """
    The Function provides just a call to flagBreaks_spektrumBased, with parameter defaults that refer to references [1].

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    thresh_rel : float, default 0.1
        Float in [0,1]. See (1) of function description above to learn more
    thresh_abs : float, default 0.01
        Float > 0. See (2) of function descritpion above to learn more.
    first_der_factor : float, default 10
        Float > 0. See (3) of function descritpion above to learn more.
    first_der_window_range : str, default '12h'
        Offset string. See (3) of function description to learn more.
    scnd_der_ratio_margin_1 : float, default 0.05
        Float in [0,1]. See (4) of function descritpion above to learn more.
    scnd_der_ratio_margin_2 : float, default 10
        Float in [0,1]. See (5) of function descritpion above to learn more.
    smooth : bool, default True
        Method for obtaining dataseries' derivatives.
        * False: Just take series step differences (default)
        * True: Smooth data with a Savitzky Golay Filter before differentiating.
    smooth_window : {None, str}, default 2
        Effective only if `smooth` = True
        Offset string. Size of the filter window, used to calculate the derivatives.
    smooth_poly_deg : int, default 2
        Effective only, if `smooth` = True
        Polynomial order, used for smoothing with savitzk golay filter.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    References
    ----------
    [1] Dorigo,W. et al.: Global Automated Quality Control of In Situ Soil Moisture
        Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
        doi:10.2136/vzj2012.0097.

    Find a brief mathematical description of the function here:

    [2] https://git.ufz.de/rdm-software/saqc/-/blob/testfuncDocs/docs/funcs
        /FormalDescriptions.md#breaks_flagspektrumbased

    """
    return breaks_flagSpektrumBased(
        data,
        field,
        flagger,
        thresh_rel=thresh_rel,
        thresh_abs=thresh_abs,
        first_der_factor=first_der_factor,
        first_der_window=first_der_window,
        scnd_der_ratio_range=scnd_der_ratio_range,
        scnd_der_ratio_thresh=scnd_der_ratio_thresh,
        smooth=smooth,
        smooth_window=smooth_window,
        smooth_poly_deg=smooth_poly_deg,
        **kwargs,
    )


@register(masking='all')
def sm_flagFrost(data, field, flagger, soil_temp_variable, window="1h", frost_thresh=0, **kwargs):

    """
    This Function is an implementation of the soil temperature based Soil Moisture flagging, as presented in
    references [1]:

    All parameters default to the values, suggested in this publication.

    Function flags Soil moisture measurements by evaluating the soil-frost-level in the moment of measurement.
    Soil temperatures below "frost_level" are regarded as denoting frozen soil state.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    soil_temp_variable : str,
        An offset string, denoting the fields name in data, that holds the data series of soil temperature values,
        the to-be-flagged values shall be checked against.
    window : str
        An offset string denoting the maximal temporal deviation, the soil frost states timestamp is allowed to have,
        relative to the data point to-be-flagged.
    frost_thresh : float
        Value level, the flagger shall check against, when evaluating soil frost level.

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    References
    ----------
    [1] Dorigo,W. et al.: Global Automated Quality Control of In Situ Soil Moisture
        Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
        doi:10.2136/vzj2012.0097.
    """

    # retrieve reference series
    refseries = data[soil_temp_variable].copy()
    ref_use = flagger.isFlagged(soil_temp_variable, flag=flagger.GOOD, comparator="==") | flagger.isFlagged(
        soil_temp_variable, flag=flagger.UNFLAGGED, comparator="=="
    )
    # drop flagged values:
    refseries = refseries[ref_use.values]
    # drop nan values from reference series, since those are values you dont want to refer to.
    refseries = refseries.dropna()
    # skip further processing if reference series is empty:
    if refseries.empty:
        return data, flagger

    refseries = refseries.reindex(data[field].dropna().index, method="nearest", tolerance=window)
    refseries = refseries[refseries < frost_thresh].index

    flagger = flagger.setFlags(field, refseries, **kwargs)
    return data, flagger


@register(masking='all')
def sm_flagPrecipitation(
    data,
    field,
    flagger,
    prec_variable,
    raise_window=None,
    sensor_depth=0,
    sensor_accuracy=0,
    soil_porosity=0,
    std_factor=2,
    std_window="24h",
    ignore_missing=False,
    **kwargs,
):

    """
    This Function is an implementation of the precipitation based Soil Moisture flagging, as presented in
    references [1].

    All parameters default to the values, suggested in this publication. (excluding porosity,sensor accuracy and
    sensor depth)


    Function flags Soil moisture measurements by flagging moisture rises that do not follow up a sufficient
    precipitation event. If measurement depth, sensor accuracy of the soil moisture sensor and the porosity of the
    surrounding soil is passed to the function, an inferior level of precipitation, that has to preceed a significant
    moisture raise within 24 hours, can be estimated. If those values are not delivered, this inferior bound is set
    to zero. In that case, any non zero precipitation count will justify any soil moisture raise.

    A data point y_t is flagged an invalid soil moisture raise, if:

    (1) y_t > y_(t-`raise_window`)
    (2) y_t - y_(t-`std_factor_range`) > `std_factor` * std(y_(t-`std_factor_range`),...,y_t)
    (3) sum(prec(t-24h),...,prec(t)) > `sensor_depth` * `sensor_accuracy` * `soil_porosity`

    NOTE1: np.nan entries in the input precipitation series will be regarded as susipicious and the test will be
    omited for every 24h interval including a np.nan entrie in the original precipitation sampling rate.
    Only entry "0" will be regarded as denoting "No Rainfall".

    NOTE2: The function wont test any values that are flagged suspicious anyway - this may change in a future version.


    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional informations related to `data`.
    prec_variable : str
        Fieldname of the precipitation meassurements column in data.
    raise_window: {None, str}, default None
        Denotes the distance to the datapoint, relatively to witch
        it is decided if the current datapoint is a raise or not. Equation [1].
        It defaults to None. When None is passed, raise_window is just the sample
        rate of the data. Any raise reference must be a multiple of the (intended)
        sample rate and below std_factor_range.
    sensor_depth : float, default 0
        Measurement depth of the soil moisture sensor, [m].
    sensor_accuracy : float, default 0
        Accuracy of the soil moisture sensor, [-].
    soil_porosity : float, default 0
        Porosity of moisture sensors surrounding soil, [-].
    std_factor : int, default 2
        The value determines by which rule it is decided, weather a raise in soil
        moisture is significant enough to trigger the flag test or not:
        Significance is assumed, if the raise is  greater then "std_factor" multiplied
        with the last 24 hours standart deviation.
    std_window: str, default '24h'
        An offset string that denotes the range over witch the standart deviation is obtained,
        to test condition [2]. (Should be a multiple of the sampling rate)
    raise_window: str
        Denotes the distance to the datapoint, relatively to witch
        it is decided if the current datapoint is a raise or not. Equation [1].
        It defaults to None. When None is passed, raise_window is just the sample
        rate of the data. Any raise reference must be a multiple of the (intended)
        sample rate and below std_factor_range.
    ignore_missing: bool, default False

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    References
    ----------
    [1] Dorigo,W. et al.: Global Automated Quality Control of In Situ Soil Moisture
        Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
        doi:10.2136/vzj2012.0097.
    """

    dataseries, moist_rate = retrieveTrustworthyOriginal(data, field, flagger)

    # data not hamronized:
    refseries = data[prec_variable].dropna()
    # abort processing if any of the measurement series has no valid entries!
    if moist_rate is np.nan:
        return data, flagger
    if refseries.empty:
        return data, flagger

    refseries = refseries.reindex(refseries.index.join(dataseries.index, how="outer"))
    # get 24 h prec. monitor
    prec_count = refseries.rolling(window="1D").sum()
    # exclude data not signifying a raise::
    if raise_window is None:
        raise_window = 1
    else:
        raise_window = int(np.ceil(pd.Timedelta(raise_window) / moist_rate))

    # first raise condition:
    raise_mask = dataseries > dataseries.shift(raise_window)

    # second raise condition:
    std_window = int(np.ceil(pd.Timedelta(std_window) / moist_rate))
    if ignore_missing:
        std_mask = dataseries.dropna().rolling(std_window).std() < (
            (dataseries - dataseries.shift(std_window)) / std_factor
        )
    else:
        std_mask = dataseries.rolling(std_window).std() < ((dataseries - dataseries.shift(std_window)) / std_factor)

    dataseries = dataseries[raise_mask & std_mask]
    invalid_indices = prec_count[dataseries.index] <= sensor_depth * sensor_accuracy * soil_porosity
    invalid_indices = invalid_indices[invalid_indices]

    # set Flags
    flagger = flagger.setFlags(field, loc=invalid_indices.index, **kwargs)
    return data, flagger


@register(masking='field')
def sm_flagConstants(
    data,
    field,
    flagger,
    window="12h",
    thresh=0.0005,
    precipitation_window="12h",
    tolerance=0.95,
    deriv_max=0.0025,
    deriv_min=0,
    max_missing=None,
    max_consec_missing=None,
    smooth_window=None,
    smooth_poly_deg=2,
    **kwargs,
):

    """
    This function flags plateaus/series of constant values in soil moisture data.

    Mentionings of "conditions" in the following explanations refer to references [2].

    The function represents a stricter version of
    constants_flagVarianceBased.

    The additional constraints (3)-(5), are designed to match the special cases of constant
    values in soil moisture measurements and basically for preceding precipitation events
    (conditions (3) and (4)) and certain plateau level (condition (5)).

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    window : str, default '12h'
        Minimum duration during which values need to identical to become plateau candidates. See condition (1)
    thresh : float, default 0.0005
        Maximum variance of a group of values to still consider them constant. See condition (2)
    precipitation_window : str, default '12h'
        See condition (3) and (4)
    tolerance : float, default 0.95
        Tolerance factor, see condition (5)
    deriv_max : float, default 0
        See condition (4)
    deriv_min : float, default 0.0025
        See condition (3)
    max_missing : {None, int}, default None
        Maximum number of missing values allowed in window, by default this condition is ignored
    max_consec_missing : {None, int}, default None
        Maximum number of consecutive missing values allowed in window, by default this condition is ignored
    smooth_window : {None, str}, default None
        Size of the smoothing window of the Savitsky-Golay filter. The default value None results in a window of two
        times the sampling rate (i.e. three values)
    smooth_poly_deg : int, default 2
        Degree of the polynomial used for smoothing with the Savitsky-Golay filter

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.

    References
    ----------
    [1] Dorigo,W. et al.: Global Automated Quality Control of In Situ Soil Moisture
        Data from the international Soil Moisture Network. 2013. Vadoze Zone J.
        doi:10.2136/vzj2012.0097.

    [2] https://git.ufz.de/rdm-software/saqc/-/edit/testfuncDocs/docs/funcs/FormalDescriptions.md#sm_flagconstants
    """

    # get plateaus:
    _, comp_flagger = constants_flagVarianceBased(
        data,
        field,
        flagger,
        window=window,
        thresh=thresh,
        max_missing=max_missing,
        max_consec_missing=max_consec_missing,
    )

    new_plateaus = (comp_flagger.getFlags(field)).eq(flagger.getFlags(field))
    # get dataseries at its sampling freq:
    dataseries, moist_rate = retrieveTrustworthyOriginal(data, field, flagger)
    # get valuse referring to dataseries:
    new_plateaus.resample(pd.Timedelta(moist_rate)).asfreq()
    # cut out test_slices for min/max derivatives condition check:
    # offset 2 periods:
    precipitation_window = int(np.ceil(pd.Timedelta(precipitation_window) / moist_rate))
    window = int(np.ceil(pd.Timedelta(window) / moist_rate))
    period_diff = precipitation_window - window
    # we cast plateua series to int - because replace has problems with replacing bools by "method".
    new_plateaus = new_plateaus.astype(int)
    # get plateau groups:
    group_counter = new_plateaus.cumsum()
    group_counter = group_counter[group_counter.diff() == 0]
    group_counter.name = "group_counter"
    plateau_groups = pd.merge(group_counter, dataseries, left_index=True, right_index=True, how="inner")
    # test mean-condition on plateau groups:
    test_barrier = tolerance * dataseries.max()
    plateau_group_drops = plateau_groups.groupby("group_counter").filter(lambda x: x[field].mean() <= test_barrier)
    # discard values that didnt pass the test from plateau candidate series:
    new_plateaus[plateau_group_drops.index] = 1

    # we extend the plateaus to cover condition testing sets
    # 1: extend backwards (with a technical "one" added):
    cond1_sets = new_plateaus.replace(1, method="bfill", limit=(precipitation_window + window))
    # 2. extend forwards:
    if period_diff > 0:
        cond1_sets = cond1_sets.replace(1, method="ffill", limit=period_diff)

    # get first derivative
    if smooth_window is None:
        smooth_window = 3 * pd.Timedelta(moist_rate)
    else:
        smooth_window = pd.Timedelta(smooth_window)
    filter_window_seconds = smooth_window.seconds
    smoothing_periods = int(np.ceil((filter_window_seconds / moist_rate.n)))
    first_derivate = savgol_filter(dataseries, window_length=smoothing_periods, polyorder=smooth_poly_deg, deriv=1,)
    first_derivate = pd.Series(data=first_derivate, index=dataseries.index, name=dataseries.name)
    # cumsumming to seperate continous plateau groups from each other:
    group_counter = cond1_sets.cumsum()
    group_counter = group_counter[group_counter.diff() == 0]
    group_counter.name = "group_counter"
    group_frame = pd.merge(group_counter, first_derivate, left_index=True, right_index=True, how="inner")
    group_frame = group_frame.groupby("group_counter")
    condition_passed = group_frame.filter(lambda x: (x[field].max() >= deriv_max) & (x[field].min() <= deriv_min))

    flagger = flagger.setFlags(field, loc=condition_passed.index, **kwargs)

    return data, flagger


@register(masking='all')
def sm_flagRandomForest(data, field, flagger, references, window_values: int, window_flags: int, path: str, **kwargs):
    """
    This Function uses pre-trained machine-learning model objects for flagging of a specific variable. The model is
    supposed to be trained using the script provided in "ressources/machine_learning/train_machine_learning.py". For
    flagging, Inputs to the model are the timeseries of the respective target at one specific sensors, the automatic
    flags that were assigned by SaQC as well as multiple reference series. Internally, context information for each
    point is gathered in form of moving windows to improve the flagging algorithm according to user input during
    model training. For the model to work, the parameters 'references', 'window_values' and 'window_flags' have to be
    set to the same values as during training.

    Parameters
    ----------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    flagger : saqc.flagger.BaseFlagger
        A flagger object, holding flags and additional Informations related to `data`.
    references : {str, List[str]}
        List or list of strings, denoting the fieldnames of the data series that should be used as reference variables
    window_values : int
        An integer, denoting the window size that is used to derive the gradients of both the field- and
        reference-series inside the moving window
    window_flags : int
        An integer, denoting the window size that is used to count the surrounding automatic flags that have been set
        before
    path : str
        A string giving the path to the respective model object, i.e. its name and
        the respective value of the grouping variable. e.g. "models/model_0.2.pkl"

    Returns
    -------
    data : dios.DictOfSeries
        A dictionary of pandas.Series, holding all the data.
    flagger : saqc.flagger.BaseFlagger
        The flagger object, holding flags and additional informations related to `data`.
        Flags values may have changed, relatively to the flagger input.
    """

    def _refCalc(reference, window_values):
        """ Helper function for calculation of moving window values """
        outdata = dios.DictOfSeries()
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

    # Function for moving window calculations
    # Create custom df for easier processing
    df = data.loc[:, [field] + references]
    # Create binary column of BAD-Flags
    df["flag_bin"] = flagger.isFlagged(field, flag=flagger.BAD, comparator="==").astype("int")

    # Add context information of flags
    # Flag at t +/-1
    df["flag_bin_t_1"] = df["flag_bin"] - df["flag_bin"].shift(1)
    df["flag_bin_t1"] = df["flag_bin"] - df["flag_bin"].shift(-1)
    # n Flags in interval t to t-window_flags
    df[f"flag_bin_t_{window_flags}"] = df["flag_bin"].rolling(window_flags + 1, center=False).sum()
    # n Flags in interval t to t+window_flags
    # forward-orientation not possible, so right-orientation on reversed data an reverse result
    df[f"flag_bin_t{window_flags}"] = df["flag_bin"].iloc[::-1].rolling(window_flags + 1, center=False).sum()[::-1]

    # TODO: dios.merge() / dios.join() ...
    # replace the following version with its DictOfSeries -> DataFrame
    # conversions as soon as merging/joining is available in dios

    # Add context information for field+references
    df = df.to_df()  # df is a dios
    for i in [field] + references:
        ref = _refCalc(reference=df[i], window_values=window_values).to_df()
        df = pd.concat([df, ref], axis=1)
    # all further actions work on pd.DataFrame. thats ok,
    # because only the df.index is used to set the actual
    # flags in the underlining dios.

    # remove NAN-rows from predictor calculation
    df = df.dropna(axis=0, how="any")
    # drop column of automatic flags at time t
    df = df.drop(columns="flag_bin")
    # Load model and predict on df:
    model = joblib.load(path)
    preds = model.predict(df)

    flag_indices = df[preds.astype("bool")].index
    flagger = flagger.setFlags(field, loc=flag_indices, **kwargs)
    return data, flagger
