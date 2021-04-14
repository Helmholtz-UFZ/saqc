import gc
from .generate_testsets import get_random_df_and_dios


def calc_mem(rows, cols, shifted=False, dtypesz=(64 / 8)):
    if shifted:
        idxsz = 8 * rows * cols
        # additional nans are inserted exactly as many as variables
        rowsz = rows * cols * dtypesz
    else:
        idxsz = 8 * rows
        rowsz = rows * dtypesz

    return idxsz + rowsz * cols


def bytes2hread(bytes):
    i = 0
    units = ["B", "kB", "MB", "GB", "TB"]
    while bytes > 1000:
        bytes /= 1024
        i += 1
        if i == 4:
            break
    return bytes, units[i]


def rows_by_time(nsec, mdays):
    """calc the number of values for one value every n seconds in m days
    :param nsec: n seconds a value occur
    :param mdays: this many days of data
    :return: rows thats needed
    """
    return int((60 / nsec) * 60 * 24 * mdays)


if __name__ == "__main__":

    # dios      - linear in rows and colums, same size for r=10,c=100 or r=100,c=10
    do_real_check = True
    cols = 10
    rows = 100000
    # rows = rows_by_time(nsec=600, mdays=365*2)

    mem = calc_mem(rows, cols, shifted=False)
    memsh = calc_mem(rows, cols, shifted=True)

    df, dios = get_random_df_and_dios(rows, cols, disalign=False, randstart=True)
    dios_mem = dios.memory_usage()
    print(f"dios:\n-----------")
    print("mem: ", *bytes2hread(dios_mem))
    print("entries:", sum([len(dios[e]) for e in dios]))
    print()

    ratio = (1 / (memsh - mem)) * dios_mem

    mem = bytes2hread(mem)
    memsh = bytes2hread(memsh)

    print("df - best case\n---------")
    print("mem: ", *mem)
    print("entries:", rows)
    print()
    print("df - worst case\n---------")
    print("mem :", *memsh)
    print("entries:", rows * cols)

    print()
    print(f"dfbest, dios, dfworst: 0%, {round(ratio, 4)*100}%, 100% ")

    if not do_real_check:
        exit(0)

    proveMeRight = False

    if proveMeRight:
        # best case
        print()
        print("best case proove")
        dfb, _ = get_random_df_and_dios(rows, cols, disalign=False, randstart=False)
        dfb.info(memory_usage="deep", verbose=False)

    print()
    print("rand start, same freq")
    df.info(memory_usage="deep", verbose=False)
    print("entries:", sum([len(df[e]) for e in df]))

    print()
    print("rand start, rand freq")
    df, _ = get_random_df_and_dios(rows, cols, disalign="random", randstart=True)
    df.info(memory_usage="deep", verbose=False)
    print("entries:", sum([len(df[e]) for e in df]))

    if proveMeRight:
        # worst case
        print()
        print("worst case proove")
        df, _ = get_random_df_and_dios(rows, cols, disalign=True, randstart=False)
        df.info(memory_usage="deep", verbose=False)

    gc.collect()
