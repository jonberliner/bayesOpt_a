from pandas import read_pickle
from numpy import isnan

def load(fname):
    assert (fname[-4:] == '.pkl' or fname[-7:] =='.pickle'), 'fname must be a pickle file'
    df = read_pickle(fname)

    crit = df.yBest.map(lambda y: not isnan(y))  # remove nan trials
    df = df[crit]
    df = df[df.status==4]  # only keep complete subjects
    df['itrial'] = df['itrial'].astype(int)
    return df


