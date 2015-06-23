from jb.jbgp_1d import K_se
from jbgp_fit import get_trialError
import numpy as np
from numpy import array as npa
from scipy.optimize import minimize_scalar
from pandas import DataFrame
import pdb

DOMAIN = np.linspace(0, 1, 1028)

def get_experimentError(dfs, lenscale):
    pdb.set_trace()
    # lenscale = dfs.LENSCALE.iat[0]
    sigvar = dfs.SIGVAR.iat[0]
    KDOMAIN = K_se(DOMAIN, DOMAIN, lenscale, sigvar)

    out = dfs.groupby('itrial').apply(lambda df0:
            get_trialError(npa(df0.xObs.iat[0]),
                           npa(df0.yObs.iat[0]),
                           df0.xDrill.iat[0],
                           df0.yDrill.iat[0],
                           DOMAIN,
                           KDOMAIN,
                           lenscale))

    out = np.vstack(out.values)

    xerr = out[:, 0]
    yerr = out[:, 1]
    xhat = out[:, 2]
    yhat = out[:, 3]

    return {'xerr': xerr,
            'yerr': yerr,
            'xhat': xhat,
            'yhat': yhat}


def bayesOpt_trialError(xObs, yObs, xSub, ySub, lenscale, sigvar):
    KDOMAIN = K_se(DOMAIN, DOMAIN, lenscale, sigvar)

    out = get_trialError(npa(xObs),
                         npa(yObs),
                         npa(xSub),
                         npa(ySub),
                         DOMAIN,
                         KDOMAIN,
                         lenscale)

    xerr = out[0]
    yerr = out[1]
    xhat = out[2]
    yhat = out[3]

    return {'xerr': xerr,
            'yerr': yerr,
            'xhat': xhat,
            'yhat': yhat}


def fit_lenscale_bayesOpt_trial(xObs, yObs, xSub, ySub, lenscale, sigvar):
    res = minimize_scalar(lambda ls: 
                    (bayesOpt_trialError(xObs, yObs, xSub, ySub, ls, sigvar)['xerr']).sum(),
                bounds=(0.0001, 0.5),
                method='bounded')
    return res


def fit_lenscale(dfs):
    res = minimize_scalar(lambda ls: 
                    (get_experimentError(dfs, ls)['xerr']).sum(),
                bounds=(0.0001, 0.5),
                method='bounded')
    return res


def fit_lenscale_analysis(df):
    assert 3 not in np.unique(df.nObs), 'remove sam3 trials first'
    # try:
    fits = []
    for wid in df.workerid.unique():
        print wid
        dfs = df[df.workerid==wid]
        res = DataFrame(fit_lenscale(dfs), index={wid})\
                .reset_index()\
                .rename(columns={'index': 'workerid'})
        fits.append(res)
    # except:
    #     print 'boo'
    return fits
    
