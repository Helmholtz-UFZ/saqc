
def doc(dc_string):
    def dc_func(meth):
        meth.__doc__ = dc_string
        return meth
    return dc_func