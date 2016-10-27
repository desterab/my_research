from beh_tools import recall_dynamics as rdf
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats as stats



def load_the_data(n_perms):

    #
    recalls = pickle.load(open(
        "/Users/khealey/Library/Mobile Documents/com~apple~CloudDocs/lab/code/experiments/Heal16implicit/HealEtal16implicit.data.pkl",
        "rb"))

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
                # pass
                continue

            # skip lists with zero recall accuracy
            rec_mat = cur_recalls.as_matrix(range(recalls.shape[1] - 2))  # format recalls matrix for use with rdf functions
            prec = rdf.prec(listlen=16, recalls=rec_mat)
            if prec <= 0.:
                continue

            # compute spc
            spc = rdf.spc(listlen=16, recalls=rec_mat, filter_ind=None)
            spc = pd.DataFrame.from_records(spc)

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

            # # compute random temporal factor
            tempf_z = rdf.relative_to_random(listlen=16, recalls=rec_mat, filter_ind=None, statistic_func=rdf.tem_fact,
                                       data_col="tf", n_perms=n_perms)
            tempf_z = pd.DataFrame.from_records(tempf_z)
            all_tf_z = tempf_z.tf.mean()

            # compute crp
            crp = rdf.crp(listlen=16, recalls=rec_mat, filter_ind=None, allow_repeats=False, exclude_op=0)
            crp = pd.DataFrame.from_records(crp)
            #
            # # compute random ctrl crp
            # crp_z = rdf.relative_to_random(listlen=16, recalls=rec_mat, filter_ind=None, statistic_func=rdf.crp,
            #                            data_col="crp", n_perms=n_perms)
            # crp_z = pd.DataFrame.from_records(crp_z)

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
            # crp['crp_z'] = crp_z.crp
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

def prec_fig(data_to_use, which_list, save_name):
    colors = ["#000000", "#000000", "#D3D3D3"]  # black and white
    sns.set_palette(colors)

    # setup the grid
    fig2 = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(1, 3)
    e1_axis = fig2.add_subplot(gs[0, 0])
    e2_axis = fig2.add_subplot(gs[0, 1:3])

    # plot prec for E1
    colors = ["#000000", "w", "#D3D3D3"]  # black and white
    sns.set_palette(colors)
    data_filter = np.logical_and(data_to_use.list == which_list, data_to_use.task_condition == 'Size')
    g = sns.barplot(x="instruction_condition", y='prec', data=data_to_use.loc[data_filter, :], order=["Explicit", "Implicit"], ax=e1_axis)
    g.patches[1].set_hatch('/')
    e1_axis.set(xlabel="Encoding Instructions", ylabel="Recall Prob.", ylim=[0., .5])

    #
    # g.patches[0].set_color("#000000")
    # g.patches[1].set_color("#808080")
    # g.patches[2].set_color("#D3D3D3")
    #
    # g.patches[3].set_edgecolor("#000000")
    # g.patches[4].set_edgecolor("#808080")
    # g.patches[5].set_edgecolor("#D3D3D3")
    #
    #
    # g.patches[3].set_hatch('/')
    # g.patches[4].set_hatch('/')
    # g.patches[5].set_hatch('/')



    # plot prec for E2
    colors = ["#000000", "#808080", "#D3D3D3"]  # black and white
    sns.set_palette(colors)
    data_filter = np.logical_and(data_to_use.list == which_list, data_to_use.instruction_condition == 'Implicit')
    g = sns.barplot(x='task_condition', y='prec', data=data_to_use.loc[data_filter, :], x_order=["Scenario", "Deep", "Relational"], ax=e2_axis)
    ax = plt.gca()
    ax.set(xlabel="Processing Task", ylabel="Recall Prob.")#, ylim=[0., .5])
    ax.legend(title='Encoding Instructions')

    plt.figure(fig2.number)
    sns.despine()



    fig2.savefig(save_name + '.pdf', bbox_inches='tight')
    plt.close(fig2)


def encoding_instruct_fig(data_to_use, which_list, save_name):
    colors = ["#000000", "#000000", "#D3D3D3"]  # black and white
    sns.set_palette(colors)

    # setup the grid
    fig2 = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(1, 2)
    crp_axis = fig2.add_subplot(gs[0, 0])
    tf_axis = fig2.add_subplot(gs[0, 1])

    # plot crps
    data_filter = data_to_use.list == which_list
    g = sns.factorplot(x="lag", y="crp", hue="instruction_condition", data=data_to_use.loc[data_filter, :], ax=crp_axis,
                   hue_order=["Explicit", "Implicit"])
    crp_axis.set(xlabel="Lag", ylabel="Cond. Resp. Prob.", ylim=[0., .2])
    crp_axis.legend().parent.get_lines()[12].set_linestyle('--')
    # crp_axis.legend().parent.get_lines()[12].set_color("#000000")
    crp_axis.legend(title='Processing Task')
    plt.figure(fig2.number)
    sns.despine()

    # plot temp factors
    colors = ["#000000", "w", "#D3D3D3"]  # black and white
    sns.set_palette(colors)
    data_filter = np.logical_and(data_to_use.list == which_list, data_to_use.lag == 0)
    g = sns.barplot(x="instruction_condition", y='all_tf_z', data=data_to_use.loc[data_filter, :], ax=tf_axis,
                order=["Explicit", "Implicit"])
    tf_axis.set(xlabel="Encoding Instructions", ylabel="Temporal Factor", ylim=[-.1, .2])
    g.patches[1].set_hatch('/')
    plt.figure(fig2.number)
    sns.despine()

    plt.suptitle("Scenario Processing Task")

    fig2.savefig(save_name + '.pdf', bbox_inches='tight')
    plt.close(fig2)



def processing_task_fig(data_to_use, which_instruction_cond, which_list, save_name):
    colors = ["#000000", "#808080", "#D3D3D3"]  # black and white
    sns.set_palette(colors)

    # setup the grid
    fig2 = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(1, 2)
    crp_axis = fig2.add_subplot(gs[0, 0])
    tf_axis = fig2.add_subplot(gs[0, 1])

    # plot crps
    data_filter = np.logical_and(data_to_use.instruction_condition == which_instruction_cond, data_to_use.list == which_list)
    sns.factorplot(x="lag", y="crp", hue="task_condition", data=data_to_use.loc[data_filter, :], ax=crp_axis,
                   hue_order=["Scenario", "Deep", "Relational"])
    crp_axis.set(xlabel="Lag", ylabel="Cond. Resp. Prob.", ylim=[0., .2])
    crp_axis.legend(title='Processing Task')
    plt.figure(fig2.number)
    sns.despine()

    # plot temp factors
    data_filter = np.logical_and(np.logical_and(data_to_use.instruction_condition == which_instruction_cond, data_to_use.list == which_list), data_to_use.lag == 0)
    sns.barplot(x="task_condition", y='all_tf_z', data=data_to_use.loc[data_filter, :], ax=tf_axis,
                order=["Scenario", "Deep", "Relational"])
    tf_axis.set(xlabel="Processing Task", ylabel="Temporal Factor", ylim=[-.1, .2])
    plt.figure(fig2.number)
    sns.despine()

    plt.suptitle(which_instruction_cond + " Encoding Instructions")

    fig2.savefig(save_name + '.pdf', bbox_inches='tight')
    plt.close(fig2)


def fig_compare_tasks(all_crps, all_spcs, which_cond=0, which_list=0, apply_perm_correction=False, print_to=None):

    if apply_perm_correction:
        which_tf_data = "all_tf_z"
        which_crp_data = "crp_z"
    else:

        which_tf_data = "all_tf"
        which_crp_data = "crp"

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

    if which_cond == 0:
        which_cond = "Explicit"
    else:
        which_cond = "Implicit"


    # bar plot of prec by task
    data_filter = np.logical_and(all_crps.instruction_condition == which_cond,
                                 np.logical_and(all_crps.lag.abs() == 5, all_crps.list == which_list))
    sns.barplot(x="task_condition", y="prec", data=all_crps.loc[data_filter, :], ax=prec_axis)
    prec_axis.set(xlabel="Processing Task", ylabel="Recall Prob.")
    sns.despine()


    # plot crps
    data_filter = np.logical_and(all_crps.instruction_condition == which_cond,
                                 np.logical_and(all_crps.lag.abs() <= 5, all_crps.list == which_list))
    sns.factorplot(x="lag", y=which_crp_data, hue="task_condition", data=all_crps.loc[data_filter, :], ax=crp_axis)
    crp_axis.set(xlabel="Lag", ylabel="Cond. Resp. Prob.")
    sns.despine()


    # examine distributions - temporal factor scores
    fig2 = plt.figure()#(figsize=(30, 10))
    gs2 = gridspec.GridSpec(2, 1)
    box_axis = fig2.add_subplot(gs[0, :])
    hist_axis = fig2.add_subplot(gs[1, :])
    data_filter = np.logical_and(all_crps.instruction_condition == which_cond,
                                 np.logical_and(all_crps.lag.abs() == 5, all_crps.list == which_list))
    sns.boxplot(x="task_condition", y=which_tf_data,  data=all_crps.loc[data_filter, :], ax=box_axis)
    sns.FacetGrid(all_crps.loc[data_filter, :], hue="task_condition").map(sns.distplot, which_tf_data, rug=True, ax=hist_axis)
    plt.figure(fig2.number)
    if print_to is not None:
        plt.savefig(print_to + "TFDIST" + '.pdf')




    # plot temp factors
    data_filter = np.logical_and(all_crps.instruction_condition == which_cond,
                                 np.logical_and(all_crps.lag.abs() == 5, all_crps.list == which_list))
    sns.barplot(x="task_condition", y=which_tf_data, data=all_crps.loc[data_filter, :], ax=tempf_axis)
    tempf_axis.set(xlabel="Processing Task", ylabel="Temporal Factor")#, ylim=[.4, .7])
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
    # for i, p in enumerate(pvals):
    #     if p >= 0.05:
    #         displaystring = r'n.s.'
    #     else:
    #         displaystring = r'*'
    #
    #     height = offset[i] + max(means)
    #     bar_centers = index[i] + np.array([0, length[i]])
    #     significance_bar(bar_centers[0], bar_centers[1], height, displaystring)


    plt.tight_layout()
    if print_to is not None:
        plt.savefig(print_to + '.pdf')


