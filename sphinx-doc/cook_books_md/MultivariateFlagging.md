# Multivariate Flagging

The tutorial aims to introduce the usage of SaQC in the context of some more complex flagging and processing techniques. 
Mainly we will see how to apply Drift Corrections onto the data and how to perform multivariate flagging.

0. * [Data Preparation](#Data-Preparation)
1. * [Drift Correction](#Drift-Correction)
2. * [Multivariate Flagging (odd Water)](#Multivariate-Flagging)

## Data preparation
* Flagging missing values via :py:func:`flagMissing <Functions.saqc.flagMissing>`.
* Flagging out of range values via :py:func:`flagRange <Functions.saqc.flagRange>`.
* Resampling the data via linear Interpolation (:py:func:`linear <Functions.saqc.linear>`).


## Drift Correction

### Exponential Drift

* The variables *SAK254* and *Turbidity* show drifting behavior originating from dirt, that accumulates on the light sensitive sensor surfaces over time.  
* The effect, the dirt accumulation has on the measurement values, is assumed to be properly described by an exponential model.
* The Sensors are cleaned periodocally, resulting in a periodical reset of the drifting effect. 
* The Dates and Times of the maintenance events are input to the :py:func:`correctDrift <Functions.saqc.correctDrift>`, that will correct the data in between any two such maintenance intervals. (Find some formal description of the process [here](sphinx-doc/misc_md/ExponentialModel.md).)

### Linear Long Time Drift

* Afterwards, there remains a long time linear Drift in the *SAK254* and *Turbidity* measurements, originating from scratches, that accumule on the sensors glass lenses over time
* The lenses are replaced periodically, resulting in a periodical reset of that long time drifting effect
* The Dates and Times of the lenses replacements are input to the :py:func:`correctDrift <Functions.saqc.correctDrift>`, that will correct the data in between any two such maintenance intervals according to the assumption of a linearly increasing bias.

### Maintenance Intervals Flagging

* The *SAK254* and *Turbidity* values, obtained while maintenance, are, of course not trustworthy, thus, all the values obtained while maintenance get flagged via the :py:func:`flagManual <Functions.saqc.flagManual>` method.
* When maintaining the *SAK254* and *Turbidity* sensors, also the *NO3* sensors get removed from the water - thus, they also have to be flagged via the :py:func:`flagManual <Functions.saqc.flagManual>` method.


## Multivariate Flagging

