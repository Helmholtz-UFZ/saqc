"""

"""


def flagByStatLowPass(field):
    """
    Flag *chunks* of length, `window`:

    1. If they excexceed `thresh` with regard to `stat`:
    2. If all (maybe overlapping) *sub-chunks* of *chunk*, with length `sub_window`,
       `excexceed `sub_thresh` with regard to `stat`:

    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-flagged.
    """
    pass
