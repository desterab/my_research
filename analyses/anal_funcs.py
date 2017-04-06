import used_recall_dynamics as rdf
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from os import system

# color palette for figures
palette =["#be5104",
"#0163c2",
"#ccdd73",
"#b15fd3",
"#b3003c"]

# params for temporal factor plots
tf_lims = [-.025, .2]
tf_col ='all_tf_z'


def load_the_data(n_perms, pool, save_name):

    #
    system('scp cbcc.psy.msu.edu:~/code/experiments/Heal16implicit/HealEtal16implicit.data.pkl \'/Users/khealey/Library/Mobile Documents/com~apple~CloudDocs/lab/code/experiments/Heal16implicit\'')
    recalls = pickle.load(open(
        "/Users/khealey/Library/Mobile Documents/com~apple~CloudDocs/lab/code/experiments/Heal16implicit/HealEtal16implicit.data.pkl",
        "rb"))

    # change conditions from numerical to verbal labelsls
    recalls.loc[recalls.task_condition == 0, 'task_condition'] = "Shoebox"
    recalls.loc[recalls.task_condition == 1, 'task_condition'] = "Movie"
    recalls.loc[recalls.task_condition == 2, 'task_condition'] = "Relational"
    recalls.loc[recalls.task_condition == 3, 'task_condition'] = "Scenario"
    recalls.loc[recalls.task_condition == 4, 'task_condition'] = "Animacy"
    recalls.loc[recalls.task_condition == 5, 'task_condition'] = "Weight"
    recalls.loc[recalls.task_condition == 6, 'task_condition'] = "Front Door"
    recalls.loc[recalls.instruction_condition == 0, 'instruction_condition'] = "Explicit"
    recalls.loc[recalls.instruction_condition == 1, 'instruction_condition'] = "Incidental"

    # loop over subjects and lists, for each isolate their data
    subjects = recalls.subject.unique()
    included_subjects = []
    n_subs = subjects.shape[0]
    si = 0.  # for a progress counter
    lists = [0]  #  doing only the first list. to do all lists change to: recalls.list.unique()
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
            incidental = cur_recalls.instruction_condition[0] == "Incidental"
            if aware.any() and incidental:
                aware_check = 1
            else:
                aware_check = 0

            # compute overall recall
            rec_mat = cur_recalls.as_matrix(range(recalls.shape[1] - 2))  # format recalls matrix for use with rdf functions
            prec = rdf.prec(listlen=16, recalls=rec_mat)
            if prec <= 0.:
                continue
            included_subjects.append(s)
            # continue


            # compute spc
            spc = rdf.spc(listlen=16, recalls=rec_mat, filter_ind=None)
            spc = pd.DataFrame.from_records(spc)

            # compute pfr
            pfr_copy = np.copy(rec_mat)
            pfr_copy[:, 1:] = np.nan  # setting all but first output to nan
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
                                       data_col="tf", n_perms=n_perms, POOL=pool)
            tempf_z = pd.DataFrame.from_records(tempf_z)
            all_tf_z = tempf_z.tf.mean()

            # compute crp
            crp = rdf.crp(listlen=16, recalls=rec_mat, filter_ind=None, allow_repeats=False, exclude_op=0)
            crp = pd.DataFrame.from_records(crp)

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
            crp["aware_check"] = pd.Series([aware_check for x in range(len(crp.index))],
                                      index=crp.index)
            all_crps = pd.concat([all_crps, crp])

            # finalize spc data: add list number, and condition ids
            spc = pd.merge(spc, pfr, how='inner', on=['serial_pos']) # merge prec and spc
            spc['subject'] = pd.Series([s for x in range(len(spc.index))], index=spc.index)
            spc['list'] = pd.Series([l for x in range(len(spc.index))], index=spc.index)
            spc['instruction_condition'] = pd.Series(
                [cur_recalls.instruction_condition[0] for x in range(len(spc.index))],
                index=spc.index)
            spc['task_condition'] = pd.Series([cur_recalls.task_condition[0] for x in range(len(spc.index))],
                                              index=spc.index)
            all_spcs = pd.concat([all_spcs, spc])


    # save data used in the anals
    e1 = recalls.task_condition == "Shoebox"
    e2 = recalls.task_condition == "Front Door"
    e3 = np.logical_and(np.in1d(recalls.task_condition, ["Weight", "Animacy", "Scenario", "Movie", "Relational"]),
                        recalls.instruction_condition == "Incidental")
    used_data = np.logical_and(np.logical_or(e1, np.logical_or(e2, e3)), recalls.list == 0)  # in a used condition and the first list
    recalls.loc[np.logical_and(np.in1d(recalls.subject, included_subjects), used_data)].to_csv(save_name)

    print "Data Loaded!"
    return all_crps, all_spcs


def encoding_instruct_fig(data_to_use, which_list, save_name):
    colors = ["#000000", "#808080"]
    sns.set_palette(colors)


    # setup the grid
    fig2 = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(1, 2)
    crp_axis = fig2.add_subplot(gs[0, 0])
    tf_axis = fig2.add_subplot(gs[0, 1])

    # plot crps
    data_filter = data_to_use.list == which_list
    g = sns.factorplot(x="lag", y="crp", hue="instruction_condition", data=data_to_use.loc[data_filter, :],
                   hue_order=["Explicit", "Incidental"], dodge=.25, units='subject', ax=crp_axis)
    crp_axis.set(xlabel="Lag", ylabel="Cond. Resp. Prob.", ylim=[0., .2], xticks=range(0, 11, 2),
                 xticklabels=range(-5, 6, 2))
    crp_axis.legend(title='Encoding Instructions', ncol=2, labelspacing=.2, handlelength=.01, loc=2)
    plt.figure(fig2.number)
    sns.despine()
    crp_axis.annotate('A.', xy=(-.175, 1), xycoords='axes fraction')

    # plot temp factors
    data_filter = np.logical_and(data_to_use.list == which_list, data_to_use.lag == 0)
    g = sns.barplot(x="instruction_condition", y=tf_col, data=data_to_use.loc[data_filter, :], order=["Explicit", "Incidental"], ax=tf_axis) #
    tf_axis.set(xlabel="Encoding Instructions", ylabel="Z(TCE)", ylim=tf_lims)
    plt.axhline(linewidth=3, linestyle='--', color='k')
    plt.figure(fig2.number)
    sns.despine()
    tf_axis.annotate('B.', xy=(-.175, 1), xycoords='axes fraction')

    fig2.savefig(save_name + '.pdf', bbox_inches='tight')
    plt.close(fig2)


def processing_task_fig(data_to_use, which_instruction_cond, which_list, save_name):
    colors = palette
    sns.set_palette(colors)

    # setup the grid
    fig2 = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(1, 2)
    crp_axis = fig2.add_subplot(gs[0, 0])
    tf_axis = fig2.add_subplot(gs[0, 1])

    # plot crps
    data_filter = np.logical_and(data_to_use.instruction_condition == which_instruction_cond, data_to_use.list == which_list)
    sns.factorplot(x="lag", y="crp", hue="task_condition", data=data_to_use.loc[data_filter, :], ax=crp_axis,
                   hue_order=["Weight", "Animacy", "Scenario", "Movie", "Relational"], dodge=.35, units='subject')
    crp_axis.set(xlabel="Lag", ylabel="Cond. Resp. Prob.", ylim=[0., .2], xticks=range(0, 11, 2), xticklabels=range(-5, 6, 2))
    crp_axis.legend(title='Judgment Task', ncol=2, labelspacing=.2, handlelength=.01, loc=2)
    plt.figure(fig2.number)
    sns.despine()
    crp_axis.annotate('A.', xy=(-.175, 1), xycoords='axes fraction')

    # plot temp factors
    data_filter = np.logical_and(np.logical_and(data_to_use.instruction_condition == which_instruction_cond, data_to_use.list == which_list), data_to_use.lag == 0)
    sns.barplot(x="task_condition", y=tf_col, data=data_to_use.loc[data_filter, :], ax=tf_axis,
                order=["Weight", "Animacy", "Scenario", "Movie", "Relational"])
    tf_axis.set(xlabel="Judgment Task", ylabel="Z(TCE)", ylim=tf_lims)
    plt.axhline(linewidth=3, linestyle='--', color='k')
    plt.figure(fig2.number)
    sns.despine()
    tf_axis.annotate('B.', xy=(-.175, 1), xycoords='axes fraction')

    fig2.savefig(save_name + '.pdf', bbox_inches='tight')
    plt.close(fig2)
