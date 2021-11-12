
Exponential Drift Model and Correction
======================================

It is assumed, that, in between maintenance events, there is a drift effect shifting the measurements in a way, that the resulting value course can be described by the exponential model $\ ``M``\ $:

$\ ``M(t, a, b, c) = a + b(e^{ct}-1)``\ $

We consider the timespan in between maintenance events to be scaled to the $\ ``[0,1]``\ $ interval.
To additionally make sure, the modeled curve can be used to calibrate the value course, we added the following two conditions.

$\ ``M(0, a, b, c) = y_0``\ $

$\ ``M(1, a, b, c) = y_1``\ $

With $\ ``y_0``\ $ denoting the mean value obtained from the first 6 meassurements directly after the last maintenance event, and $\ ``y_1``\ $ denoting the mean over the 6 meassurements, directly preceeding the beginning of the next maintenance event.

Solving the equation, one obtains the one-parameter Model:

$\ ``M_{drift}(t, c) = y_0 + ( \frac{y_1 - y_0}{e^c - 1} ) (e^{ct} - 1)``\ $

For every datachunk in between maintenance events.

After having found the parameter $\ ``c^*``\ $, that minimizes the squared residues between data and drift model, the correction is performed by bending the fitted curve, $\ ``M_{drift}(t, c^*)``\ $, in a way, that it matches $\ ``y_2``\ $ at $\ ``t=1``\ $ (,with $\ ``y_2``\ $ being the mean value observed directly after the end of the next maintenance event).
This bended curve is given by:

$\ ``M_{shift}(t, c^{*}) = M(t, y_0,  \frac{y_1 - y_0}{e^c - 1} , c^*)``\ $

the new values $\ ``y_{shifted}``\ $ are computed via:

$\ ``y_{shifted} = y + M_{shift} - M_{drift}``\ $
