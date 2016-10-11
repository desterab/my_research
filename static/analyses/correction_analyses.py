from contiguity_model import *
from beh_tools import recall_dynamics as rdf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats as stats
import timeit
import shelve
from joblib import Parallel, delayed
import multiprocessing






# model parameters
params = {"n_lists": 100,
          "n_items": 16,
          "p_rec": 0.5,
          'spc': [.8, .75, .7, .65, .6, .55, .55, .55, .55, .55, .6, .65, .7, .75, .85, .9],
          "model_contiguity": True,
          "a_pos": 0,
          "a_neg": 0.5,
          "asym": 1.2}

# different spcs
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
# [.8, .75, .7, .65, .6, .55, .55, .55, .55, .55, .6, .65, .7, .75, .85, .9]
# [1, .95, .9, .85, .8, .003, .003, .003, .003, .003, .003, .003, .85, .9, .95, 1]

#  run a bunch of subjects
n_ss = 50
n_perms = 1

# at a bunch of levels of prec
these_precs = np.array([.75, .5, .25])

conditions = [("lo_p", "med_c"), ("hi_p", "med_c")]

cond_params = {"lo_p": .25,
               "med_p": .5,
               "hi_p": .75,
               "lo_c": .5,
               "med_c": 1.0,
               "hi_c": 1.5}

asym_factor = 1


prec_results = pd.DataFrame()
tempf_results = pd.DataFrame()
tempf_z_results = pd.DataFrame()
spc_results = pd.DataFrame()
crp_results = pd.DataFrame()
crp_z_results = pd.DataFrame()
true_results = pd.DataFrame()
for cond_i in conditions:
    print cond_i
    params['p_rec'] = cond_params[cond_i[0]]
    params['a_pos'] = cond_params[cond_i[1]]
    params["a_neg"] = params['a_pos'] * asym_factor


    for s_i in range(n_ss):

        # run the model
        recalls = random_transition(params)

        # compute actual prec
        prec = rdf.prec(listlen=16, recalls=recalls)
        prec = pd.DataFrame([[s_i, params['p_rec'], prec]], columns=['subject', 'param_prec', 'observed_prec'])
        prec_results = pd.concat([prec_results, prec])

        # compute spc
        spc = rdf.spc(listlen=16, recalls=recalls)
        spc = pd.DataFrame.from_records(spc)
        spc['subject'] = pd.Series([s_i for x in range(len(spc.index))], index=spc.index)
        spc['param_prec'] = pd.Series([params['p_rec'] for x in range(len(spc.index))], index=spc.index)
        spc_results = pd.concat([spc_results, spc])

        # compute crp
        crp = rdf.crp(listlen=16, recalls=recalls)
        crp = pd.DataFrame.from_records(crp)
        crp['subject'] = pd.Series([s_i for x in range(len(crp.index))], index=crp.index)
        crp['param_prec'] = pd.Series([params['p_rec'] for x in range(len(crp.index))], index=crp.index)
        crp_results = pd.concat([crp_results, crp])

        # compute random ctrl crp
        crp_z = rdf.relative_to_random(listlen=16, recalls=recalls, filter_ind=None, statistic_func=rdf.crp,
                                       data_col="numer", n_perms=n_perms)
        crp_z = pd.DataFrame.from_records(crp_z)
        crp_z['subject'] = pd.Series([s_i for x in range(len(crp.index))], index=crp_z.index)
        crp_z['param_prec'] = pd.Series([params['p_rec'] for x in range(len(crp_z.index))], index=crp_z.index)
        crp_z_results = pd.concat([crp_z_results, crp_z])

        # compute temporal factor
        tempf = rdf.tem_fact(listlen=16, recalls=recalls)
        tempf = np.nanmean(tempf['tf'])
        tempf = pd.DataFrame([[s_i, params['p_rec'], tempf]], columns=['subject', 'param_prec', 'observed_tempf'])
        tempf_results = pd.concat([tempf_results, tempf])

        # # compute random temporal factor
        # tempf_z = rdf.relative_to_random(listlen=16, recalls=recalls, filter_ind=None, statistic_func=rdf.tem_fact,
        #                                  data_col="tf", n_perms=n_perms)
        # tempf_z = pd.DataFrame.from_records(tempf_z)
        # tempf_z = tempf_z.tf.mean()
        # tempf_z = pd.DataFrame([[s_i, params['p_rec'], tempf_z]], columns=['subject', 'param_prec', 'observed_tempf_z'])
        # tempf_z_results  = pd.concat([tempf_z_results, tempf_z])

        # true contiguity implied by model param
        pos_lag_probs = params["asym"] * (np.arange(params["n_items"] - 1) + 1) ** -params["a_pos"]
        neg_lag_probs = (np.arange(params["n_items"] - 1) + 1) ** -params["a_neg"]
        lag_probs = np.concatenate((neg_lag_probs[::-1], pos_lag_probs))
        lag_probs = pd.DataFrame(lag_probs, columns=['b'])
        lags = np.concatenate((np.arange(-params["n_items"] + 1, 0), np.arange(1, params["n_items"])))
        lag_probs['param_prec'] = pd.Series([params['p_rec'] for x in range(len(lag_probs.index))], index=lag_probs.index)
        lag_probs['lag'] = pd.Series([x for x in lags], index=lag_probs.index)
        true_results = pd.concat([true_results, pd.DataFrame([[np.nan, 0, params['p_rec']]], columns=['b', 'lag', 'param_prec']),
                                  lag_probs])




# save everything

# my_shelf = shelve.open(filename,'n') # 'n' for new
# for key in dir():
#     try:
#         my_shelf[key] = globals()[key]
#     except TypeError:
#         #
#         # __builtins__, my_shelf, and imported modules can not be shelved.
#         #
#         print('ERROR shelving: {0}'.format(key))
# my_shelf.close()
#
#
#
# filename='shelve.out'
# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     globals()[key]=my_shelf[key]
# my_shelf.close()

# sns.factorplot(x='param_prec', y='observed_tempf', data=tempf_results)
#
# sns.factorplot(x='param_prec', y='observed_tempf_z', data=tempf_z_results)
# # plt.savefig('tempf' + '.pdf')

data_fiter = true_results.lag.abs() <= 5
sns.factorplot(x="lag", y='b', hue="param_prec", data=true_results.loc[data_fiter, :])

data_fiter = crp_results.lag.abs() <= 5
sns.factorplot(x="lag", y='crp', hue="param_prec", data=crp_results.loc[data_fiter, :])

data_fiter = crp_z_results.lag.abs() <= 5
sns.factorplot(x="lag", y='crp', hue="param_prec", data=crp_z_results.loc[data_fiter, :])






sns.factorplot(x="serial_pos", y='prec', hue="param_prec", data=spc_results)

x=1


lo = crp_results.loc[crp_results.param_prec == .25].groupby('lag', as_index=False).mean()
hi = crp_results.loc[crp_results.param_prec == .75].groupby('lag', as_index=False).mean()

plt.plot(hi.crp.values[16:16+6] / hi.crp.values[17:17+6])
plt.plot(lo.crp.values[16:16 + 6] / lo.crp.values[17:17 + 6])






## sractch

def plag(lag):

    a = .8
    b = -1.2

    return a * lag**b






#
# ##### tcm version
# #  run a bunch of subjects
# n_ss = 1000
#
# # at a bunch of levels of prec
# these_precs = np.linspace(start=0., stop=1., num=20)
#
# # model parameters
# params = {"n_lists": 100,
#           "n_items": 16,
#           "n_recall_attempts": 20,
#           "b_enc": 0.561,
#           "b_rec": 0.375,
#           "p_rec": 0.5}
#
#
# prec_results = pd.DataFrame()
# tempf_results = pd.DataFrame()
# for p_i, cur_prec in enumerate(these_precs):
#     print p_i
#     for s_i in range(n_ss):
#         params['p_rec'] = cur_prec
#         recalls = run_model(params)
#
#         # compute actual prec
#         prec = rdf.prec(listlen=16, recalls=recalls)
#         prec = pd.DataFrame([[s_i, cur_prec, prec]], columns=['subject', 'param_prec', 'observed_prec'])
#         prec_results = pd.concat([prec_results, prec])
#
#         # compute temporal factor
#         tempf = rdf.tem_fact(listlen=16, recalls=recalls)
#         tempf = np.nanmean(tempf['tf'])
#         tempf = pd.DataFrame([[s_i, cur_prec, tempf]], columns=['subject', 'param_prec', 'observed_tempf'])
#         tempf_results = pd.concat([tempf_results, tempf])
#
#
#
#
# sns.factorplot(x='param_prec', y='observed_tempf', data=tempf_results)
# plt.savefig('tempf' + '.pdf')