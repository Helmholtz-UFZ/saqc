.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

Writing Functions
=================

Writing non-standard functions
------------------------------

When implementing non-standard functions, i.e. all function not decorated with ``flagging``, some special care is
needed to comply to the standard ``SaQC`` behaviour. The following passages guide you through the jungle of
``register`` arguments and there semantics.

Masking
"""""""
TODO

Demasking
"""""""""
TODO

History squeezing
"""""""""""""""""
TODO

Target handling
"""""""""""""""
Functions decorated with ``register(handles_target=True)`` are fully in charge to implement the source-target
workflow. That means they need to handle the following cases:

- Existing ``target``: one ore more of the given targets might already exist in ``data`` and ``flags``.
  There is (currently) no hard rule how we expect functions to handle this case, the default functions
  however drop pre-existing ``target`` with:

  .. code-block:: python

     from saqc.funcs import dropField

     data, flags = dropField(data=data, fags=flags, field=field, target=target)

- Non-existing ``target``: we stick to the idea, that non existing targets will be handled
  gracefully. Usually that implies, that after a function returns all names given in ``target`` exist
  in ``data`` as well as in ``flags``. The easiest solution here is to create an explicit copy:

  .. code-block:: python

     from saqc.funcs import copyField

     data, flags = copyField(data=data, flags=flags, field=field, target=target)

  However, such a direct mapping/lineage between ``field`` and ``target`` is not always meaningful. Usually
  the function itself either provides new ``flags`` or new ``data``, but we need to also provide values for
  for the respective other structure. For ``data`` we default to a series with all ``np.nan`` values, e.g.

  .. code-block:: python

     data[target] = pd.Series(np.nan, index=data[field].index)
     
  and empty ``flags``, e.g.

  .. code-block:: python

     from saqc.core import History
     flags.history[target] = saqc.core.history.History(data[target].index)
  
- Incompatible lengths of ``field`` and ``target``: Certain functions might have specific mapping
  needs, some multivariate functions, for example, might map exactly three ``fields`` to one ``target``,
  or need exactly on ``target`` for every field. If this is the case, such invriants should be checked
  accordingly.

- Incompatible indices for ``field`` and ``target``: Depending on the function, differing indices for
  ``field`` and ``target`` might or might not impose problems for the implementation. If not coded carefully,
  chances are however high, that this might cause breakage. It is therefore recommended to check this
  accordingly.
  
