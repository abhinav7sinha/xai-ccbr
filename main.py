import pandas as pd
import util

if __name__=='__main__':
    input_file='data/travel_cb.txt'

    util=util.Util()
    df=util.build_df(input_file)
    print(df.head())