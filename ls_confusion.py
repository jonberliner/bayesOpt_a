from jb.jbdrill import jbload
from jb import jbgp
from numpy import array as a
from jb.jbdrill.jbgp_fit import get_experimentError
from numpy import linspace, unique, zeros, zeros_like
from time import time

import pdb


NROUND = 200
## LOCAL DB (used if dbs recently synced with sync_mysql_aws2local.sh)
DB_URL = 'mysql://root:PASSWORD@127.0.0.1/myexp'
## AWS DB
# DB_URL = 'mysql://jsb4:PASSWORD@mydb.c4dh2hic3vxp.us-east-1.rds.amazonaws.com:3306/myexp'

FINISHED_STATUSES = [3,4,5,7]   # psiturk markers for a completed experiment
# trials that do not return true for all functions in criterion will not be used
CRITERION = [lambda df: 'round' in df,  # gets rid of non-my-exp trials (e.g. psiturk gunk)
            lambda df: df['round'] > 0 and df['round'] <= NROUND,  # same as above
            lambda df: df['status'] in FINISHED_STATUSES]  # taking complete exps only

# get experiments data
df = jbload.noChoice_exp0(DB_URL, CRITERION)  #load noChoice experiment

workers = unique(df.workerid)
trials = unique(df.round)
nw = len(workers)
nt = len(trials)

LSPOOL = 2.**-a([2., 4., 6.])
DOMAIN = linspace(0, 1, 1028)
out = []
for w in workers:
    print str(w)+' of '+str(nw)
    workertime = time()
    dfw = df[df.workerid==w]
    lsw = dfw.lenscale.iat[0]
    svw = dfw.sigvar.iat[0]
    nv2w = dfw.noisevar2.iat[0]

    # make rectangular array for cython
    nObs = a([len(obsX0) for obsX0 in dfw.obsX])
    xObs = zeros([nt, max(nObs)])
    yObs = zeros_like(xObs)

    xDrill = dfw.drillX.values
    yDrill = dfw.drillX.values

    lspoolerrs = a([get_experimentError(xObs,
                                        yObs,
                                        nObs,
                                        xDrill,
                                        yDrill,
                                        ls0,
                                        DOMAIN)[:,0].mean()
                    for ls0 in LSPOOL])


    mut = jbgp.conditioned_mu(DOMAIN, txObs, tyObs, tls, svw, nv2w)
    out.append({'err': err,
                'best_fit_ls': LSPOOL[lspoolerrs.argmin()],
                'exp_ls': lsw,
                'worker': w,
                'lspoolerrs': lspoolerrs})
    print 'time: '+str(time()-workertime)

