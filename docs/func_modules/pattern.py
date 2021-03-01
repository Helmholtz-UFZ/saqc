"""

"""
def flagPatternByDTW(field):
    """
    Pattern recognition via wavelets.
    
    The steps are:
     1. work on chunks returned by a moving window
     2. each chunk is compared to the given pattern, using the wavelet algorithm as presented in [1]
     3. if the compared chunk is equal to the given pattern it gets flagged
    
    Parameters
    ----------
    
    field : str
        The fieldname of the data column, you want to correct.
    """
    pass


def flagPatternByWavelet(field):
    """
    Pattern Recognition via Dynamic Time Warping.
    
    The steps are:
     1. work on chunks returned by a moving window
     2. each chunk is compared to the given pattern, using the dynamic time warping algorithm as presented in [1]
     3. if the compared chunk is equal to the given pattern it gets flagged
    
    Parameters
    ----------
    
    field : str
        The fieldname of the data column, you want to correct.
    """
    pass

