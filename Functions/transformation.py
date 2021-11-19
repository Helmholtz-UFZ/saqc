"""

"""
def transform(field, func, freq, target):
    """
    Function to transform data columns with a transformation that maps series onto series of the same length.
    
    Note, that flags get preserved.
    
    Parameters
    ----------
    field : str
        The fieldname of the column, holding the data-to-be-transformed.
    func : Callable[{pd.Series, np.array}, np.array]
        Function to transform data[field] with.
    freq : {None, float, str}, default None
        Determines the segmentation of the data into partitions, the transformation is applied on individually
    
        * ``np.inf``: Apply transformation on whole data set at once
        * ``x`` > 0 : Apply transformation on successive data chunks of periods length ``x``
        * Offset String : Apply transformation on successive partitions of temporal extension matching the passed offset
          string
    target : str or None, default None
        Write the result of the processing to the new variable ``target``. Must not already exist.
    
    """
    pass

