from builtins import range

from dios import *
import numpy as np

if __name__ == "__main__":
    dios_options[OptsFields.mixed_itype_warn_policy] = Opts.itype_warn
    print(dios_options)

    df = pd.DataFrame(columns=range(1000))
    pd.Series()
    # print(df)
    # exit(99)

    # dios_options[OptsFields.disp_max_cols] = 5
    # dios_options[OptsFields.disp_max_rows] = 100
    dios_options[OptsFields.disp_min_rows] = 50
    # dios_options[OptsFields.dios_repr] = Opts.repr_aligned

    n = 10
    d = DictOfSeries(
        dict(
            l=pd.Series(0, index=range(0, 30)),
            # i123=pd.Series(dtype='O'),
            a=pd.Series(1, index=range(0, n)),
            nan=pd.Series(np.nan, index=range(3, n + 3)),
            b=pd.Series(2, index=range(0, n * 2, 2)),
            c=pd.Series(3, index=range(n, n * 2)),
            d=pd.Series(4, index=range(-n // 2, n // 2)),
            # z=pd.Series([1, 2, 3], index=list("abc"))
        )
    )

    def f(s):
        sec = 10 ** 9
        s.index = pd.to_datetime(s.index * sec)
        return s

    dd = d.apply(f)
    print(d)

    # print(d.to_df())
    # print(pd.options.display.max_rows)
    # print(d.to_str(col_delim=' | ', col_space=20, header_delim='0123456789'))
    # print(d.to_str(col_delim=' | ', col_space=20, max_cols=4 ))
    di = DictOfSeries(columns=[])
    print(di)
    # print(DictOfSeries(data=1, columns=['a']))
