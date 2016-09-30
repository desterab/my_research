from beh_tools import recall_dynamics as rdf
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats as stats



def load_the_data():

    # scp cbcc.psy.msu.edu:/home/khealey/code/experiments/Heal16implicit/HealEtal16implicit.data.pkl /Users/khealey/code/experiments/Heal16implicit/
    recalls = pickle.load(open("/Users/khealey/code/experiments/Heal16implicit/HealEtal16implicit.data.pkl", "rb"))

    # loop over subjects and lists, for each isolate their data
    subjects = recalls.subject.unique()
    n_subs = subjects.shape[0]
    si = 0. # for a progress counter
    lists = recalls.list.unique()
    all_crps = pd.DataFrame()
    all_spcs = pd.DataFrame()
    for s in subjects:
        si += 1
        print si / n_subs * 100.
        for l in lists:

            # get the data for just this list
            s_filter = recalls.subject == s
            l_filter = recalls.list == l
            cur_recalls = recalls.loc[s_filter & l_filter, :]

            # skip if there were no recalls
            if cur_recalls.shape[0] == 0:
                continue

            # skip lists if subject reported being aware
            aware = cur_recalls.aware[0].values == 'yes'
            implicit = cur_recalls.instruction_condition[0] == 1
            if aware.any() and implicit:
                continue

            # skip lists with zero recall accuracy
            rec_mat = cur_recalls.as_matrix(range(recalls.shape[1] - 2))  # format recalls matrix for use with rdf functions
            prec = rdf.prec(listlen=16, recalls=rec_mat)
            if prec <= 0:
                continue

            # compute spc
            spc = rdf.spc(listlen=16, recalls=rec_mat, filter_ind=None)
            spc = pd.DataFrame.from_records(spc) # make all but first col 0, then spc as pfr and add as col to all_spcs

            # compute pfr
            pfr_copy = np.copy(rec_mat)
            pfr_copy[:, 1:] = np.nan # setting all but first output to nan
            pfr = rdf.spc(listlen=16, recalls=pfr_copy, filter_ind=None)
            pfr = pd.DataFrame.from_records(pfr)
            pfr = pfr.rename(columns={'prec': 'pfr'})  # rename the data col to avoid confusion with spc


            # compute temporal factor, getting average separately for positive and negative lags
            tempf = rdf.tem_fact(listlen=16, recalls=rec_mat, filter_ind=None)
            tempf = pd.DataFrame.from_records(tempf)
            pos_tf = tempf.tf[tempf.lag > 0].mean()
            neg_tf = tempf.tf[tempf.lag < 0].mean()
            all_tf = tempf.tf.mean()

            # compute random temporal factor
            tempf_z = rdf.relative_to_random(listlen=16, recalls=rec_mat, filter_ind=None, statistic_func=rdf.tem_fact,
                                       data_col="tf", n_perms=1000)
            tempf_z = pd.DataFrame.from_records(tempf_z)
            all_tf_z = tempf_z.tf.mean()

            # compute crp
            crp = rdf.crp(listlen=16, recalls=rec_mat, filter_ind=None, allow_repeats=False, exclude_op=0)
            crp = pd.DataFrame.from_records(crp)

            # compute random ctrl crp
            crp_z = rdf.relative_to_random(listlen=16, recalls=rec_mat, filter_ind=None, statistic_func=rdf.crp,
                                       data_col="crp", n_perms=1000)
            crp_z = pd.DataFrame.from_records(crp_z)

            # fanilize crp data: add list number, and condition ids
            crp['subject'] = pd.Series([s for x in range(len(crp.index))], index=crp.index)
            crp['list'] = pd.Series([l for x in range(len(crp.index))], index=crp.index)
            crp['instruction_condition'] = pd.Series([cur_recalls.instruction_condition[0] for x in range(len(crp.index))],
                                                     index=crp.index)
            crp['task_condition'] = pd.Series([cur_recalls.task_condition[0] for x in range(len(crp.index))],
                                              index=crp.index)
            crp['prec'] = pd.Series([prec for x in range(len(crp.index))],
                                    index=crp.index)  # todo: this needlessly makes a copy of prec for each lag... this seems like a xarray issue
            crp['pos_tf'] = pd.Series([pos_tf for x in range(len(crp.index))],
                                    index=crp.index)
            crp['neg_tf'] = pd.Series([neg_tf for x in range(len(crp.index))],
                                    index=crp.index)
            crp['all_tf'] = pd.Series([all_tf for x in range(len(crp.index))],
                                    index=crp.index)
            crp['all_tf_z'] = pd.Series([all_tf_z for x in range(len(crp.index))],
                                      index=crp.index)
            crp['crp_z'] = crp_z.crp
            all_crps = pd.concat([all_crps, crp])

            # fanilize spc data: add list number, and condition ids
            spc = pd.merge(spc, pfr, how='inner', on=['serial_pos']) # merge prec and spc
            spc['subject'] = pd.Series([s for x in range(len(spc.index))], index=spc.index)
            spc['list'] = pd.Series([l for x in range(len(spc.index))], index=spc.index)
            spc['instruction_condition'] = pd.Series(
                [cur_recalls.instruction_condition[0] for x in range(len(spc.index))],
                index=spc.index)
            spc['task_condition'] = pd.Series([cur_recalls.task_condition[0] for x in range(len(spc.index))],
                                              index=spc.index)
            all_spcs = pd.concat([all_spcs, spc])

    print "Data Loaded!"
    return all_crps, all_spcs


########figures###########

def fig_compare_tasks(all_crps, all_spcs, which_cond=0, which_list=0, print_to=None):

    # overall figure style
    # sns.set_style("white")
    sns.set_style("whitegrid")
    # sns.set_style("ticks")
    sns.set_context("talk")
    colors = ["#000000", "#808080", "#D3D3D3"]  # black and white
    sns.set_palette(colors)
    # sns.set_palette(sns.dark_palette("grey", n_colors=3))


    ######### Figure 1
    # which_cond = 1 # implicit = 1, explicit = 0

    #
    # # setup the grid
    fig1 = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(3, 5)

    prec_axis = fig1.add_subplot(gs[0, 0])
    spc_axis = fig1.add_subplot(gs[1, 0])
    pfr_axis = fig1.add_subplot(gs[2, 0])
    crp_axis = fig1.add_subplot(gs[:, 1:3])
    tempf_axis = fig1.add_subplot(gs[:, 3:])

    # plot spcs
    data_filter = np.logical_and(all_spcs.instruction_condition == which_cond, all_spcs.list == which_list)
    sns.factorplot(x="serial_pos", y="prec", hue="task_condition", data=all_spcs.loc[data_filter, :], ax=spc_axis)
    spc_axis.set(xlabel="Serial Position", ylabel="Recall Prob.")
    sns.despine()

    # plot spcs
    data_filter = np.logical_and(all_spcs.instruction_condition == which_cond, all_spcs.list == which_list)
    sns.factorplot(x="serial_pos", y="pfr", hue="task_condition", data=all_spcs.loc[data_filter, :], ax=pfr_axis)
    pfr_axis.set(xlabel="Serial Position", ylabel="Prob. First Recall")
    sns.despine()

    # bar plot of prec by task
    data_filter = np.logical_and(all_crps.instruction_condition == which_cond,
                                 np.logical_and(all_crps.lag.abs() == 5, all_crps.list == which_list))
    sns.barplot(x="task_condition", y="prec", data=all_crps.loc[data_filter, :], ax=prec_axis)
    prec_axis.set(xlabel="Processing Task", ylabel="Recall Prob.")
    sns.despine()

    # plot crps
    data_filter = np.logical_and(all_crps.instruction_condition == which_cond,
                                 np.logical_and(all_crps.lag.abs() <= 5, all_crps.list == which_list))
    sns.factorplot(x="lag", y="crp_z", hue="task_condition", data=all_crps.loc[data_filter, :], ax=crp_axis)
    crp_axis.set(xlabel="Lag", ylabel="Cond. Resp. Prob.")
    sns.despine()



    # plot temp factors
    data_filter = np.logical_and(all_crps.instruction_condition == which_cond,
                                 np.logical_and(all_crps.lag.abs() == 5, all_crps.list == which_list))
    sns.barplot(x="task_condition", y="all_tf", data=all_crps.loc[data_filter, :], ax=tempf_axis)
    tempf_axis.set(xlabel="Processing Task", ylabel="Temporal Factor", ylim=[.4, .7])
    sns.despine()

    t01 = stats.ttest_ind(a=all_crps.loc[np.logical_and(data_filter, all_crps.task_condition == 0), :].all_tf,
                        b=all_crps.loc[np.logical_and(data_filter, all_crps.task_condition == 1), :].all_tf,
                        nan_policy="omit")
    t02 = stats.ttest_ind(a=all_crps.loc[np.logical_and(data_filter, all_crps.task_condition == 0), :].all_tf,
                        b=all_crps.loc[np.logical_and(data_filter, all_crps.task_condition == 2), :].all_tf,
                        nan_policy="omit")
    t12 = stats.ttest_ind(a=all_crps.loc[np.logical_and(data_filter, all_crps.task_condition == 1), :].all_tf,
                        b=all_crps.loc[np.logical_and(data_filter, all_crps.task_condition == 2), :].all_tf,
                        nan_policy="omit")

    # barplot just opened a new figure window, so switch back to fig1 before doing anything else
    plt.figure(fig1.number)
    from matplotlib.markers import TICKDOWN
    def significance_bar(start, end, height, displaystring, linewidth=1.2, markersize=8, boxpad=0.3, fontsize=15,
                         color='k'):
        # draw a line with downticks at the ends
        plt.plot([start, end], [height] * 2, '-', color=color, lw=linewidth, marker=TICKDOWN, markeredgewidth=linewidth,
                 markersize=markersize)
        # draw the text with a bounding box covering up the line
        plt.text(0.5 * (start + end), height, displaystring, ha='center', va='center',
                 bbox=dict(facecolor='1.', edgecolor='none', boxstyle='Square,pad=' + str(boxpad)), size=fontsize)

    pvals = [t01.pvalue, t02.pvalue, t12.pvalue]
    index = [0, 0, 1]  # where to put the bars
    length = [.9, 2, .9]
    means = [all_crps.loc[np.logical_and(data_filter, all_crps.task_condition == 0), :].all_tf.mean(),
             all_crps.loc[np.logical_and(data_filter, all_crps.task_condition == 1), :].all_tf.mean(),
             all_crps.loc[np.logical_and(data_filter, all_crps.task_condition == 2), :].all_tf.mean()]
    offset = [.025, .05, .025]
    for i, p in enumerate(pvals):
        if p >= 0.05:
            displaystring = r'n.s.'
        else:
            displaystring = r'*'

        height = offset[i] + max(means)
        bar_centers = index[i] + np.array([0, length[i]])
        significance_bar(bar_centers[0], bar_centers[1], height, displaystring)


    plt.tight_layout()
    if print_to is not None:
        plt.savefig(print_to + '.pdf')


