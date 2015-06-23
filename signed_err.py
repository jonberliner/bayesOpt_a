from pandas import DataFrame, concat, read_pickle
from jbgp_fit import get_trialError
from jb.jbgp_1d import K_se
from numpy import linspace
from numpy import array as npa
from numba import jit

DOMAIN = linspace(0, 1, 1028)
X = DOMAIN
SIGVAR = 1.
df = read_pickle('df_noChoice_sam3.pkl')
df3 = df[df.nObs!=3]

@jit
def trial_wrapper(dft, ls):
    if dft.shape[0] != 1:
        # errpool.append(dft.copy())
        # ii = randint(dft.shape[0])
        # dft = dft.iloc[ii:ii+1]
        trialError = None
    else:
        KX = K_se(X, X, ls, SIGVAR)
        xObs = npa(dft.xObs.iat[0])
        yObs = npa(dft.yObs.iat[0])
        xDrill = dft.xDrill.iat[0]
        yDrill = dft.yDrill.iat[0]
        trialError = get_trialError(xObs, yObs, xDrill, yDrill, X, KX, ls)

