from contiguity_model import *
from beh_tools import recall_dynamics as rdf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats as stats



# model parameters
params = {"n_lists": 100,
          "n_items": 16,
          "p_rec": 0.5,
          'spc': [.8, .75, .7, .65, .6, .55, .55, .55, .55, .55, .6, .65, .7, .75, .85, .9],
          "model_contiguity": True,
          "a_pos": 0.9,
          "a_neg": 0.5,
          "asym": 1.2}

# different spcs
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
# [.8, .75, .7, .65, .6, .55, .55, .55, .55, .55, .6, .65, .7, .75, .85, .9]
# [1, .95, .9, .85, .8, .003, .003, .003, .003, .003, .003, .003, .85, .9, .95, 1]

#  run a bunch of subjects
n_ss = 10

# at a bunch of levels of prec
these_precs = np.array([.75, .5, .25])

conditions = [("lo_p", "med_c"), ("med_p", "med_c"), ("hi_p", "med_c")]

cond_params = {"lo_p": .25,
               "med_p": .5,
               "hi_p": .75,
               "lo_c": .5,
               "med_c": 1,
               "hi_c": 1.5}

asym_factor = 1


prec_results = pd.DataFrame()
tempf_results = pd.DataFrame()
spc_results = pd.DataFrame()
crp_results = pd.DataFrame()
for p_i, cur_prec in enumerate(these_precs):
    print p_i
    for s_i in range(n_ss):
        params['p_rec'] = cur_prec
        recalls = random_transition(params)

        # compute actual prec
        prec = rdf.prec(listlen=16, recalls=recalls)
        prec = pd.DataFrame([[s_i, cur_prec, prec]], columns=['subject', 'param_prec', 'observed_prec'])
        prec_results = pd.concat([prec_results, prec])

        # compute spc
        spc = rdf.spc(listlen=16, recalls=recalls)
        spc = pd.DataFrame.from_records(spc)
        spc['subject'] = pd.Series([s_i for x in range(len(spc.index))], index=spc.index)
        spc['param_prec'] = pd.Series([p_i for x in range(len(spc.index))], index=spc.index)
        spc_results = pd.concat([spc_results, spc])

        # compute crp
        crp = rdf.crp(listlen=16, recalls=recalls)
        crp = pd.DataFrame.from_records(crp)
        crp['subject'] = pd.Series([s_i for x in range(len(crp.index))], index=crp.index)
        crp['param_prec'] = pd.Series([p_i for x in range(len(crp.index))], index=crp.index)
        crp_results = pd.concat([crp_results, crp])

        # compute temporal factor
        tempf = rdf.tem_fact(listlen=16, recalls=recalls)
        tempf = np.nanmean(tempf['tf'])
        tempf = pd.DataFrame([[s_i, cur_prec, tempf]], columns=['subject', 'param_prec', 'observed_tempf'])
        tempf_results = pd.concat([tempf_results, tempf])

sns.factorplot(x='param_prec', y='observed_tempf', data=tempf_results)
# plt.savefig('tempf' + '.pdf')

data_fiter = crp_results.lag.abs() <= 5
sns.factorplot(x="lag", y='crp', hue="param_prec", data=crp_results.loc[data_fiter, :])

sns.factorplot(x="serial_pos", y='prec', hue="param_prec", data=spc_results)












##### tcm version
#  run a bunch of subjects
n_ss = 1000

# at a bunch of levels of prec
these_precs = np.linspace(start=0., stop=1., num=20)

# model parameters
params = {"n_lists": 100,
          "n_items": 16,
          "n_recall_attempts": 20,
          "b_enc": 0.561,
          "b_rec": 0.375,
          "p_rec": 0.5}


prec_results = pd.DataFrame()
tempf_results = pd.DataFrame()
for p_i, cur_prec in enumerate(these_precs):
    print p_i
    for s_i in range(n_ss):
        params['p_rec'] = cur_prec
        recalls = run_model(params)

        # compute actual prec
        prec = rdf.prec(listlen=16, recalls=recalls)
        prec = pd.DataFrame([[s_i, cur_prec, prec]], columns=['subject', 'param_prec', 'observed_prec'])
        prec_results = pd.concat([prec_results, prec])

        # compute temporal factor
        tempf = rdf.tem_fact(listlen=16, recalls=recalls)
        tempf = np.nanmean(tempf['tf'])
        tempf = pd.DataFrame([[s_i, cur_prec, tempf]], columns=['subject', 'param_prec', 'observed_tempf'])
        tempf_results = pd.concat([tempf_results, tempf])




sns.factorplot(x='param_prec', y='observed_tempf', data=tempf_results)
plt.savefig('tempf' + '.pdf')