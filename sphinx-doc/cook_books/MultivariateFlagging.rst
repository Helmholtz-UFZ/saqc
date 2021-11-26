
Multivariate Flagging
=====================

The tutorial aims to introduce the usage of SaQC in the context of some more complex flagging and processing techniques. 
Mainly we will see how to apply Drift Corrections onto the data and how to perform multivariate flagging.


#. :ref:`Data Preparation <cook_books/MultivariateFlagging:Data Preparation>`

#. :ref:`Drift Correction <cook_books/MultivariateFlagging:Drift Correction>`

#. `Multivariate Flagging (odd Water) <#Multivariate-Flagging>`_

Data preparation
----------------


* Flagging missing values via :py:func:`flagMissing <Functions.saqc.flagMissing>`.
* Flagging out of range values via :py:func:`flagRange <Functions.saqc.flagRange>`.
* Flagging values, where the Specific Conductance (\ *K25*\ ) drops down to near zero. (via :py:func:`flagGeneric <Functions.saqc.flag>`)
* Resampling the data via linear Interpolation (:py:func:`linear <Functions.saqc.linear>`).

Drift Correction
----------------

Exponential Drift
^^^^^^^^^^^^^^^^^


* The variables *SAK254* and *Turbidity* show drifting behavior originating from dirt, that accumulates on the light sensitive sensor surfaces over time.  
* The effect, the dirt accumulation has on the measurement values, is assumed to be properly described by an exponential model.
* The Sensors are cleaned periodocally, resulting in a periodical reset of the drifting effect. 
* The Dates and Times of the maintenance events are input to the :py:func:`correctDrift <Functions.saqc.correctDrift>`, that will correct the data in between any two such maintenance intervals. (Find some formal description of the process :doc:`here <../misc/ExponentialModel>`.)

Linear Long Time Drift
^^^^^^^^^^^^^^^^^^^^^^


* Afterwards, there remains a long time linear Drift in the *SAK254* and *Turbidity* measurements, originating from scratches, that accumule on the sensors glass lenses over time
* The lenses are replaced periodically, resulting in a periodical reset of that long time drifting effect
* The Dates and Times of the lenses replacements are input to the :py:func:`correctDrift <Functions.saqc.correctDrift>`, that will correct the data in between any two such maintenance intervals according to the assumption of a linearly increasing bias.

Maintenance Intervals Flagging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* The *SAK254* and *Turbidity* values, obtained while maintenance, are, of course not trustworthy, thus, all the values obtained while maintenance get flagged via the :py:func:`flagManual <Functions.saqc.flagManual>` method.
* When maintaining the *SAK254* and *Turbidity* sensors, also the *NO3* sensors get removed from the water - thus, they also have to be flagged via the :py:func:`flagManual <Functions.saqc.flagManual>` method.

Multivariate Flagging
---------------------

Basically following the *oddWater* procedure, as suggested in *Talagala, P.D. et al (2019): A Feature-Based Procedure for Detecting Technical Outliers in Water-Quality Data From In Situ Sensors. Water Ressources Research, 55(11), 8547-8568.*


* Variables *SAK254*\ , *Turbidity*\ , *Pegel*\ , *NO3N*\ , *WaterTemp* and *pH* get transformed to comparable scales
* We are obtaining nearest neighbor scores and assigign those to a new variable, via :py:func:`assignKNNScores <Functions.saqc.assignKNNScores>`.
* We are applying the *STRAY* Algorithm to find the cut_off points for the scores, above which values qualify as outliers. (:py:func:`flagByStray <Functions.saqc.flagByStray>`)
* We project the calculated flags onto the input variables via :py:func:`assignKNNScore <Functions.saqc.assignKNNScore>`.

Postprocessing
--------------


* (Flags reduction onto subspaces)
* Back projection of calculated flags from resampled Data onto original data via :py:func: ``mapToOriginal <Functions.saqc.mapToOriginal>``
