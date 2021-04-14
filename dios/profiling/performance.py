import pandas as pd
import numpy as np
import time
from .generate_testsets import get_testset, var_prefix

profile_assignment = False

idx = pd.IndexSlice
rows = 0

fir = ["var", "ts", "ass"]
sec = ["df", "a", "b", "dios"]
timingsdf = pd.DataFrame(columns=pd.MultiIndex.from_product([fir, sec]))


def df_timmings(df, t0, t1, v1, v2):
    _t0 = time.time()
    a = df.loc[t0:t1, :]
    _t1 = time.time()
    b = df.loc[:, v1]
    _t2 = time.time()
    if profile_assignment:
        df.loc[t0:t1, v1] = df.loc[t0:t1, v1] * 1111
    _t3 = time.time()

    timingsdf.at[rows, ("ts", "df")] += _t1 - _t0
    timingsdf.at[rows, ("var", "df")] += _t2 - _t1
    timingsdf.at[rows, ("ass", "df")] += _t3 - _t2
    return a, b, df


def a_timings(df, t0, t1, v1, v2):
    _t0 = time.time()
    a = df.loc[t0:t1, :]
    _t1 = time.time()
    b = df.loc[:, v1]
    _t2 = time.time()
    if profile_assignment:
        df.loc[t0:t1, v1] = df.loc[t0:t1, v1] * 1111
    _t3 = time.time()

    timingsdf.at[rows, ("ts", "a")] += _t1 - _t0
    timingsdf.at[rows, ("var", "a")] += _t2 - _t1
    timingsdf.at[rows, ("ass", "a")] += _t3 - _t2
    return a, b, df


def b_timings(df, t0, t1, v1, v2):
    _t0 = time.time()
    a = df.loc[:, t0:t1]
    _t1 = time.time()
    b = df.loc[v1, :]
    _t2 = time.time()
    if profile_assignment:
        df.loc[v1, t0:t1] = df.loc[v1, t0:t1] * 1111
    _t3 = time.time()

    timingsdf.at[rows, ("ts", "b")] += _t1 - _t0
    timingsdf.at[rows, ("var", "b")] += _t2 - _t1
    timingsdf.at[rows, ("ass", "b")] += _t3 - _t2
    return a, b, df


def dios_timings(dios, t0, t1, v1, v2):
    _t0 = time.time()
    a = dios.loc[t0:t1, :]
    _t1 = time.time()
    b = dios.loc[:, v1]
    _t2 = time.time()
    if profile_assignment:
        dios.loc[t0:t1, v1] = dios.loc[t0:t1, v1] * 1111
    _t3 = time.time()

    timingsdf.at[rows, ("ts", "dios")] += _t1 - _t0
    timingsdf.at[rows, ("var", "dios")] += _t2 - _t1
    timingsdf.at[rows, ("ass", "dios")] += _t3 - _t2
    return a, b, dios


def gen_random_timestamps(m, M):
    r = (M - m) * (np.random.randint(10, 90) + np.random.random()) * 0.01
    a, b = m + r, M - r
    return min(a, b), max(a, b)


def find_index_range(obj):
    min_ = None
    max_ = None
    for r in obj:
        m = obj[r].index.min()
        M = obj[r].index.max()
        try:
            min_ = min(min_, m)
            max_ = max(max_, M)
        except TypeError:
            min_ = m
            max_ = M
    return min_, max_


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # do not touch
    rows = 1

    # max increase of of rows
    # 1 = 10 # 2 = 100 # .... # 5 = 100'000
    iterations = 5
    runs = 1
    cols = 10

    profile_assignment = True

    # which to calc and plot
    use_df = False
    use_a = True
    use_b = True
    use_dios = True

    # plot options
    normalize_to_df = True
    plot_xlog = True
    plot_ylog = True

    # ########################

    v1 = "var1"
    v2 = "var2"
    for i in range(iterations):
        rows *= 10

        timingsdf.loc[rows] = (0,) * len(timingsdf.columns)

        df, a, b, dios = get_testset(rows, cols)
        t0, t4 = find_index_range(df)

        if use_df or normalize_to_df:
            for r in range(runs):
                t1, t2 = gen_random_timestamps(t0, t4)
                vr1 = var_prefix + str(np.random.randint(0, cols))
                df_timmings(df, t1, t2, vr1, None)

        if use_a:
            for r in range(runs):
                t1, t2 = gen_random_timestamps(t0, t4)
                vr1 = var_prefix + str(np.random.randint(0, cols))
                a_timings(a, t1, t2, vr1, None)

        if use_b:
            for r in range(runs):
                t1, t2 = gen_random_timestamps(t0, t4)
                vr1 = var_prefix + str(np.random.randint(0, cols))
                b_timings(b, t1, t2, vr1, None)

        if use_dios:
            for r in range(runs):
                t1, t2 = gen_random_timestamps(t0, t4)
                vr1 = var_prefix + str(np.random.randint(0, cols))
                dios_timings(dios, t1, t2, vr1, None)

    # calc the average
    timingsdf /= runs

    pd.set_option("display.max_columns", 100)

    df = timingsdf
    if not profile_assignment:
        df.drop(labels="ass", axis=1, level=0, inplace=True)
    print("timings:")
    print(df)
    df = df.swaplevel(axis=1)
    if normalize_to_df:
        a = df.loc[:, "a"] / df.loc[:, "df"]
        b = df.loc[:, "b"] / df.loc[:, "df"]
        c = df.loc[:, "df"] / df.loc[:, "df"]
        d = df.loc[:, "dios"] / df.loc[:, "df"]
        df.loc[:, "a"] = a.values
        df.loc[:, "b"] = b.values
        df.loc[:, "df"] = c.values
        df.loc[:, "dios"] = d.values
        all = df.copy()
        all.swaplevel(axis=1)
        print("\n\ndiff:")
        print(all)

    a = df.loc[:, ("a", slice(None))]
    b = df.loc[:, ("b", slice(None))]
    dios = df.loc[:, ("dios", slice(None))]
    df = df.loc[:, ("df", slice(None))]

    ax = plt.gca()
    ax.set_title(f"avg of: {runs} runs, columns: {cols}")

    if use_df:
        df.plot(logy=plot_ylog, logx=plot_xlog, linestyle="-", ax=ax)
    if use_a:
        a.plot(logy=plot_ylog, logx=plot_xlog, linestyle="--", ax=ax)
    if use_b:
        b.plot(logy=plot_ylog, logx=plot_xlog, linestyle=":", ax=ax)
    if use_dios:
        dios.plot(logy=plot_ylog, logx=plot_xlog, linestyle="-.", ax=ax)

    plt.show()
