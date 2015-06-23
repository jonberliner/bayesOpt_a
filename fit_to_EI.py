import jb.jbgp_1d as gp
import jb.jbaqusitionFunctions as acq
import numpy as np
from numpy import array as npa
from numpy.random import RandomState
from scipy.optimize import minimize_scalar
from pandas import DataFrame, concat
from fit_lenscale import fit_lenscale
import prep_bayesOpt
import pdb

rngseed = None  # if want to replicate exactly
if rngseed: rng = RandomState(rngseed)
else: rng = RandomState()


df = prep_bayesOpt.load('df_bayesOpt_e_062215.pickle')
NROW, NCOL = df.shape

idf0 = rng.randint(NROW)
trialSeries = df.iloc[idf0]


def ei_experiment(df):
    subjectdfs = []
    for iwid, wid in enumerate(df.workerid.unique()):
        try:
            print str(iwid) + ' of ' + str(len(df.workerid.unique()))
            dfs = df[df.workerid==wid]
            subjectdfs.append(ei_subject(dfs))
        except:
            msg = ''.join([wid, ' failed'])
            print Warning(msg)
    return concat(subjectdfs)


def ei_subject(dfs):
    assert len(dfs.workerid.unique())==1
    trialdfs = []
    for itrial in dfs.itrial.unique():
        dft = dfs[dfs.itrial==itrial]
        assert dft.shape[0] == 1
        trialSeries = dft.iloc[0]
        trialdfs.append(ei_trialSeries(trialSeries))
    return concat(trialdfs)


def ei_trialSeries(trialSeries):
    wid = trialSeries.workerid
    condition = trialSeries.condition
    counterbalance = trialSeries.counterbalance
    lenscale = trialSeries.LENSCALE
    sigvar = trialSeries.SIGVAR
    noisevar2 = 1e-7
    DOMAIN = np.linspace(0, 1, 1028)
    KDOMAIN = gp.K_se(DOMAIN, DOMAIN, lenscale, sigvar)


    # trial by trial fits to expected improvement
    nPassiveObs = len(trialSeries.xPassiveObs)
    nActiveObs = len(trialSeries.xActiveObs)
    xActive = trialSeries.xActiveObs
    yActive = trialSeries.yActiveObs
    xPassive = trialSeries.xPassiveObs
    yPassive = trialSeries.yPassiveObs

    minidicts = []
    for iActive in xrange(nActiveObs):
        # get active obs to this point
        xAct = xActive[:iActive]
        yAct = yActive[:iActive]
        # combine all obs seen to this point
        xObs = npa(xAct + xPassive)
        yObs = npa(yAct + yPassive)
        xBest = xObs.max()
        yBest = yObs.max()
        # USING TRUE LENSCALE
        # get posterior
        mu = gp.conditioned_mu(DOMAIN, xObs, yObs, lenscale, sigvar, noisevar2)
        cm = gp.conditioned_covmat(DOMAIN, KDOMAIN, xObs, lenscale, sigvar, noisevar2)
        sd = np.diag(cm)
        # get EI guess
        eiout = acq.EI(yBest, mu, sd, DOMAIN)
        xEI = eiout['xmax']
        yEI = eiout['fmax']
        # get subject guess
        xSub = xActive[iActive]
        ySub = yActive[iActive]
        # compare
        xDiff = xSub - xEI
        # store
        minidicts.append({'xEI': xEI,
                        'yEI': yEI,
                        'xSub': xSub,
                        'ySub': ySub,
                        'xDiff': xDiff,
                        'iActive': iActive,
                        'xAct': xAct,
                        'yAct': yAct,
                        'xPassive': xPassive,
                        'yPassive': yPassive,
                        'exp_ls': lenscale,
                        'xAct': xAct,
                        'yAct': yAct,
                        'workerid': wid,
                        'condition': condition,
                        'counterbalance': counterbalance})
    return DataFrame(minidicts)
