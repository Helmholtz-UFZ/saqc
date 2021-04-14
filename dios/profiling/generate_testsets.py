import time

import pandas as pd
import numpy as np
import datetime as dt
from ..dios import DictOfSeries
import pickle
import os

var_prefix = "var"


def _gen_testset(rowsz, colsz, freq="1min", disalign=True, randstart=True):
    df = pd.DataFrame()
    dos = DictOfSeries()
    start = dt.datetime.strptime("2000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    times = pd.date_range(periods=rowsz, start=start, freq=freq)

    frequ = freq.strip("0123456789")
    freqv = int(freq[: -len(frequ)])

    for i in range(colsz):

        if randstart:
            # generate random startpoint for each series
            r = str(np.random.randint(int(rowsz * 0.05), int(rowsz * 0.6) + 2)) + frequ
            st = start + pd.Timedelta(r)
            times = pd.date_range(periods=rowsz, start=st, freq=freq)

        if disalign:
            if disalign == "random":
                r = np.random.randint(1, i + 2)
            else:
                # total disalign
                r = i
            times += pd.Timedelta(f"{r}ns")

        d = np.random.randint(1, 9, rowsz)
        v = f"var{i}"
        tmp = pd.DataFrame(index=times, data=d, columns=[v])
        df = pd.merge(df, tmp, left_index=True, right_index=True, how="outer")
        dos[v] = tmp.squeeze().copy()

    return df, dos


def get_random_df_and_dios(rowsz, colsz, freq="1min", disalign=True, randstart=True):
    df, _, _, dios, *_ = get_testset(
        rowsz, colsz, freq=freq, disalign=disalign, randstart=randstart
    )
    return df, dios


def get_testset(
    rows,
    cols,
    freq="1s",
    disalign=True,
    randstart=True,
    storagedir=None,
    noresult=False,
):
    if storagedir is None:
        storagedir = os.path.dirname(__file__)
        storagedir = os.path.join(storagedir, "testsets")

    fname = f"set_f{freq}_d{disalign}_r{randstart}_dim{rows}x{cols}.pkl"
    fpath = os.path.join(storagedir, fname)

    # try to get pickled data
    try:
        with open(fpath, "rb") as fh:
            if noresult:
                return
            tup = pickle.load(fh)

            # file/data was present
            return tup
    except (pickle.UnpicklingError, FileNotFoundError):
        pass

    # generate testset(s)
    df, dios = _gen_testset(
        rowsz=rows, colsz=cols, freq=freq, disalign=disalign, randstart=randstart
    )
    df = df.sort_index(axis=0, level=0)
    df_type_a = df.copy().stack(dropna=False).sort_index(axis=0, level=0).copy()
    df_type_b = df.copy().unstack().sort_index(axis=0, level=0).copy()
    tup = df, df_type_a, df_type_b, dios

    # store testsets
    with open(fpath, "wb") as fh:
        pickle.dump(tup, fh)

    if noresult:
        return

    return tup


def gen_all(rrange, crange):
    for r in rrange:
        for c in crange:
            print(r, " x ", c)
            t0 = time.time()
            get_testset(r, c, noresult=True)
            t1 = time.time()
            print(t1 - t0)


if __name__ == "__main__":
    # import time
    #
    # t0 = time.time()
    # for i in range(7):
    #     get_testset(10**i, 10)
    # t1 = time.time()
    # print(t1-t0)

    rr = [10 ** r for r in range(1, 6)]
    c = range(10, 60, 10)
    gen_all(rr, c)
