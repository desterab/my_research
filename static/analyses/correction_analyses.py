from contiguity_model import *
from beh_tools import recall_dynamics as rdf
import pandas as pd
import xarray as xr
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
params = {"n_lists": 500,
          "n_items": 16,
          'spc': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ],
          "model_contiguity": True}

# different spcs
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
# [.8, .75, .7, .65, .6, .55, .55, .55, .55, .55, .6, .65, .7, .75, .85, .9]
# [1, .95, .9, .85, .8, .003, .003, .003, .003, .003, .003, .003, .85, .9, .95, 1]

# [.8, .65, .35, .1, .001, .0001, .00001, .00001, .0001, .001, .1, .35, .65, .8, .9, 1.]

#  run a bunch of subjects
n_ss = 10
# n_perms = 1

# at a bunch of levels of prec
these_precs = np.array([.75, .5, .25])

conditions = [("lo_p", "med_c", "spurious"), ("hi_p", "med_c", "spurious"),
              ("lo_p", "lo_c", "hit"), ("hi_p", "hi_c", "hit"),
              ("lo_p", "hi_c", "mis"), ("hi_p", "lo_c", "mis")]

cond_params = {"lo_p": .25,
               "med_p": .5,
               "hi_p": .75,
               "lo_c": .5,
               "med_c": 1.0,
               "hi_c": 1.5}

decay_asym_factor = 1
base_asym_factor = .8


prec_results = pd.DataFrame()
tempf_results = pd.DataFrame()
tempf_z_results = pd.DataFrame()
spc_results = pd.DataFrame()
crp_results = pd.DataFrame()
crp_z_results = pd.DataFrame()
true_results = pd.DataFrame()
first_loop = True
for cond_i in conditions:
    print cond_i
    params['p_rec'] = cond_params[cond_i[0]]
    params['a_pos'] = cond_params[cond_i[1]] * decay_asym_factor
    params["a_neg"] = cond_params[cond_i[1]]
    params["b_pos"] = 1
    params["b_neg"] = 1 * base_asym_factor
    cond_name = cond_i[2]


    for s_i in range(n_ss):

        # run the model
        recalls = random_recalls(params)

        # true contiguity implied by model param --- which is p(lag) = b * f(lag)

        # f_lag part of crp
        pos_lag_probs = ((np.arange(params["n_items"] - 1) + 1) ** -params["a_pos"])
        neg_lag_probs = (np.arange(params["n_items"] - 1) + 1) ** -params["a_neg"]
        f_lag = np.concatenate((neg_lag_probs[::-1], np.concatenate((np.empty(1) * np.nan, pos_lag_probs))))
        f_lag = f_lag / np.nansum(f_lag)

        # b part of crp
        pos_lag_b_probs = params["b_pos"] * np.ones(params["n_items"]-1)
        neg_lag_b_probs = params["b_neg"] * np.ones(params["n_items"]-1)
        p_j = np.concatenate((neg_lag_b_probs[::-1], np.concatenate((np.empty(1) * np.nan, pos_lag_b_probs))))

        # full crp
        p_lag = p_j * f_lag

        # put it all in a dataframe
        f_lag = pd.DataFrame({"f_lag": f_lag, "p_j": p_j, "p_lag": p_lag})
        lags = np.concatenate((np.arange(-params["n_items"] + 1, 0), np.zeros(1), np.arange(1, params["n_items"])))
        f_lag['param_prec'] = pd.Series([params['p_rec'] for x in range(len(f_lag.index))], index=f_lag.index)
        f_lag['lag'] = pd.Series([x for x in lags], index=f_lag.index)
        f_lag['cond'] = pd.Series([cond_name for x in lags], index=f_lag.index)
        true_results = pd.concat([true_results, pd.DataFrame([[np.nan, 0, params['p_rec']]], columns=['b', 'lag', 'param_prec']),
                                  f_lag])


        # compute actual prec
        prec = rdf.prec(listlen=16, recalls=recalls)
        prec = pd.DataFrame([[s_i, params['p_rec'], prec, cond_name]], columns=['subject', 'param_prec', 'observed_prec', 'cond'])
        prec_results = pd.concat([prec_results, prec])

        # compute spc
        spc = rdf.spc(listlen=16, recalls=recalls)
        spc = pd.DataFrame.from_records(spc)
        spc['subject'] = pd.Series([s_i for x in range(len(spc.index))], index=spc.index)
        spc['param_prec'] = pd.Series([params['p_rec'] for x in range(len(spc.index))], index=spc.index)
        spc['cond'] = pd.Series([cond_name for x in range(len(spc.index))], index=spc.index)
        spc_results = pd.concat([spc_results, spc])

        # compute crp
        crp = rdf.crp(listlen=16, recalls=recalls)
        crp = pd.DataFrame.from_records(crp)
        crp['subject'] = pd.Series([s_i for x in range(len(crp.index))], index=crp.index)
        crp['param_prec'] = pd.Series([params['p_rec'] for x in range(len(crp.index))], index=crp.index)
        crp['cond'] = pd.Series([cond_name for x in range(len(crp.index))], index=crp.index)
        crp_results = pd.concat([crp_results, crp])

        # compute crp_graident
        crp_graident = rdf.crp_graident(listlen=16, recalls=recalls)
        crp_graident = crp_graident.rename({"row": "subject"}) # rename the default row dimension to subject, because here they are subjects
        crp_graident["subject"].values = [s_i]
        crp_graident["param_prec"] = ("subject", [params['p_rec']])
        crp_graident["cond"] = ("subject", [cond_name])
        if first_loop:
            gradient = crp_graident
            first_loop = False
        else:
            gradient = xr.concat([gradient, crp_graident], dim="subject")





        #
        # # compute random ctrl crp
        # crp_z = rdf.relative_to_random(listlen=16, recalls=recalls, filter_ind=None, statistic_func=rdf.crp,
        #                                data_col="numer", n_perms=n_perms)
        # crp_z = pd.DataFrame.from_records(crp_z)
        # crp_z['subject'] = pd.Series([s_i for x in range(len(crp.index))], index=crp_z.index)
        # crp_z['param_prec'] = pd.Series([params['p_rec'] for x in range(len(crp_z.index))], index=crp_z.index)
        # crp_z_results = pd.concat([crp_z_results, crp_z])
        #
        # # compute temporal factor
        # tempf = rdf.tem_fact(listlen=16, recalls=recalls)
        # tempf = np.nanmean(tempf['tf'])
        # tempf = pd.DataFrame([[s_i, params['p_rec'], tempf]], columns=['subject', 'param_prec', 'observed_tempf'])
        # tempf_results = pd.concat([tempf_results, tempf])

        # # compute random temporal factor
        # tempf_z = rdf.relative_to_random(listlen=16, recalls=recalls, filter_ind=None, statistic_func=rdf.tem_fact,
        #                                  data_col="tf", n_perms=n_perms)
        # tempf_z = pd.DataFrame.from_records(tempf_z)
        # tempf_z = tempf_z.tf.mean()
        # tempf_z = pd.DataFrame([[s_i, params['p_rec'], tempf_z]], columns=['subject', 'param_prec', 'observed_tempf_z'])
        # tempf_z_results  = pd.concat([tempf_z_results, tempf_z])

sns.set_style("whitegrid")
# sns.set_style("ticks")
sns.set_context("talk")
colors = ["#000000", "#808080", "#D3D3D3"]  # black and white
sns.set_palette(colors)





#########F2
fig2 = plt.figure(figsize=(30, 10))
gs = gridspec.GridSpec(1, 3)
f_lag_ax = fig2.add_subplot(gs[0, 0])
p_j_ax = fig2.add_subplot(gs[0, 1])
p_lag_ax = fig2.add_subplot(gs[0, 2])

data_fiter = np.logical_and(true_results.lag.abs() <= 5, true_results.cond == 'spurious')
sns.factorplot(x="lag", y='f_lag', hue="param_prec", data=true_results.loc[data_fiter, :], jitter=True, ax=f_lag_ax)
sns.despine()

data_fiter = np.logical_and(true_results.lag.abs() <= 5, true_results.cond == 'spurious')
sns.factorplot(x="lag", y='p_j', hue="param_prec", data=true_results.loc[data_fiter, :], ax=p_j_ax)
sns.despine()

data_fiter = np.logical_and(true_results.lag.abs() <= 5, true_results.cond == 'spurious')
sns.factorplot(x="lag", y='p_lag', hue="param_prec", data=true_results.loc[data_fiter, :], ax=p_lag_ax)
sns.despine()

fig2.savefig("F2" + '.pdf')
plt.close('all')




###########F3
conds = ['spurious', 'hit', 'mis']
for cond_name in conds:

    # # setup the grid
    fig1 = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(1, 3)

    spc_axis = fig1.add_subplot(gs[0, 0])
    true_axis = fig1.add_subplot(gs[0, 1])
    measured_axis = fig1.add_subplot(gs[0, 2])

    data_fiter = spc_results.cond == cond_name
    sns.factorplot(x="serial_pos", y='prec', hue="param_prec", data=spc_results.loc[data_fiter, :], ax=spc_axis)
    sns.despine()

    data_fiter = np.logical_and(true_results.lag.abs() <= 5, true_results.cond == cond_name)
    sns.factorplot(x="lag", y='f_lag', hue="param_prec", data=true_results.loc[data_fiter, :], ax=true_axis)
    sns.despine()

    data_fiter = np.logical_and(crp_results.lag.abs() <= 5, crp_results.cond == cond_name)
    sns.factorplot(x="lag", y='crp', hue="param_prec", data=crp_results.loc[data_fiter, :], ax=measured_axis)
    sns.despine()




    grp = crp_results["crp"].groupby(crp_results["lag"]).mean()
    # compute the percent change from lag i to i+1
    max_lag = grp['lag'].max()
    min_lag = grp['lag'].min()
    lag_i = range(min_lag + 1, 0) + range(0, max_lag) # from |1| to |max_lag|-1
    lag_i_plus_1 = range(min_lag, -1) + range(1, max_lag + 1) # from |2| to |max_lag|
    return crp_array.sel(lag=lag_i) / crp_array.sel(lag=lag_i_plus_1).values



    fig1.savefig('crp_' + cond_name + '.pdf')
    plt.close('all')


########F4

# convert gradient to pandas for plotting
grad_df = gradient.to_dataframe('grad').reset_index()

conds = ['spurious', 'hit', 'mis']
for cond_name in conds:

    # # setup the grid
    fig1 = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(1, 3)

    spc_axis = fig1.add_subplot(gs[0, 0])
    true_axis = fig1.add_subplot(gs[0, 1])
    measured_axis = fig1.add_subplot(gs[0, 2])

    data_fiter = spc_results.cond == cond_name
    sns.factorplot(x="serial_pos", y='prec', hue="param_prec", data=spc_results.loc[data_fiter, :], ax=spc_axis)
    sns.despine()

    data_fiter = np.logical_and(true_results.lag.abs() <= 5, true_results.cond == cond_name)
    sns.factorplot(x="lag", y='f_lag', hue="param_prec", data=true_results.loc[data_fiter, :], ax=true_axis)
    sns.despine()

    data_fiter = np.logical_and(grad_df.lag.abs() <= 5, grad_df.cond == cond_name)
    sns.factorplot(x="lag", y='grad', hue="param_prec", data=grad_df.loc[data_fiter, :], ax=measured_axis)
    sns.despine()

    fig1.savefig("grad_" + cond_name + '.pdf')
    plt.close('all')










