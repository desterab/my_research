import used_recall_dynamics as rdf
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from joblib import Parallel
import multiprocessing
from cbcc_tools.data_munging import spell_checker
from pyxdameraulevenshtein import damerau_levenshtein_distance_ndarray

# figure style params
sns.set_style("ticks")
sns.set_context("talk", font_scale=2.5, rc={"lines.linewidth": 4})
colors = ["#000000", "#808080", "#D3D3D3"]  # black and white
sns.set_palette(colors)

# color palette for figures
palette =["#be5104",
"#0163c2",
"#ccdd73",
"#b15fd3",
"#b3003c"]

# params for temporal factor plots
tf_lims = [-.025, .2]
tf_col ='all_tf_z'


def make_psiturk_recall_matrix(data, remake_data_file, dict_path, save_file):
    if os.path.isfile(save_file + ".pkl") and not remake_data_file:
        recalls = pickle.load(open(save_file + ".pkl", "rb"))
        return recalls

    recalls = pd.DataFrame()

    # load the webesters dict
    dict_file = open(dict_path, "r")
    dictionary = dict_file.read().split()
    dict_file.close()

    # first normalize the recalls, pools, and dictionary to get rid of obvious typos like caps and spaces and to make
    # everything lowercase
    data.word = data.word.str.replace(" ", "").str.lower()
    data.response = data.response.str.replace(" ", "").str.lower()
    dictionary = [word.lower().replace(" ", "") for word in dictionary]

    # loop over subjects, for each isolate their data
    subjects = data.uniqueid.unique()
    n_ss = len(subjects)
    loop = 0.
    for s in subjects:
        loop += 1.
        print(loop/n_ss*100)
        s_filter = data.uniqueid == s
        recalls_filter = data.phase == 'recall'
        study_filter = data.phase == 'study'
        awareness_filter = data.aware_question == 'awarenesscheck'
        aware = data.loc[s_filter & awareness_filter, 'aware_ans']
        cur_recalls = data.loc[s_filter & recalls_filter, ['list', 'response', 'instruction_condition','task_condition', 'recall_instruction_condition']]
        cur_items = data.loc[s_filter & study_filter, ['list', 'word']]

        if cur_recalls.recall_instruction_condition.isnull().all():
            print ('we are only doing E4 conditions skipping this one!!')
            continue

        # somehow, there seems to be some uniqueid's that are have two of each list.... just move on if that is the case
        if cur_items.shape[0] != 32 and cur_items.shape[0] != 16:
            print "DUP!"
            continue

        # loop over this subject's lists
        lists = cur_recalls.list.unique()
        for l in lists:
            list_filter = cur_recalls.list == l
            recalled_this_list = cur_recalls.loc[list_filter, :]

            # loop over recalls in this list and for each find its serial position or mark as intrusion then put it all
            # in a dataframe
            sp = []
            op = []
            for index, recall in recalled_this_list.iterrows():
                sp.append(which_item(recall, cur_items.loc[cur_items.list <= recall.list], dictionary))
                op.append(int(np.where(recalled_this_list.index == index)[0]))  # the output position
                # print (recall)

            # we need to add the subject id and conditions to the beginning of the line
            sp.extend((s, recall.list, recall.instruction_condition, recall.task_condition, recall.recall_instruction_condition, aware)) #sp.insert(0, s)
            op.extend(('subject', 'list', 'instruction_condition', 'task_condition', 'recall_instruction_condition', 'aware' )) #op.insert(0, 'subject')
            recalls = recalls.append(pd.DataFrame([sp], columns=tuple(op)))

    recalls.set_index('subject')
    recalls.to_pickle(save_file + ".pkl")
    return recalls


def load_the_data(n_perms, remake_data_file, recalls_file, save_name):

    if os.path.isfile(save_name + "all_crps.pkl") and not remake_data_file:
        all_crps = pickle.load(open(save_name + "all_crps.pkl", "rb"))
        return all_crps
    else:
        num_cores = multiprocessing.cpu_count() / 2
        with Parallel(n_jobs=num_cores, verbose=0) as POOL:
            # os.system('scp cbcc.psy.msu.edu:~/code/experiments/Heal16implicit/HealEtal16implicit.data.pkl \'/Users/khealey/Library/Mobile Documents/com~apple~CloudDocs/lab/code/experiments/Heal16implicit\'')
            recalls = pickle.load(open(
                recalls_file,
                "rb"))

            # loop over subjects and lists, for each isolate their data
            subjects = recalls.subject.unique()
            included_subjects = []
            n_subs = subjects.shape[0]
            si = 0.  # for a progress counter
            lists = [0]  #  doing only the first list. to do all lists change to: recalls.list.unique()
            all_crps = pd.DataFrame()
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
                    incidental = cur_recalls.instruction_condition[0] == 1
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

                    # compute random temporal factor
                    tempf_z = rdf.relative_to_random(listlen=16, recalls=rec_mat, filter_ind=None, statistic_func=rdf.tem_fact,
                                               data_col="tf", n_perms=n_perms, POOL=POOL)
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
                    crp['recall_instruction_condition'] = pd.Series([cur_recalls.recall_instruction_condition[0] for x in range(len(crp.index))],
                                                             index=crp.index)
                    crp['task_condition'] = pd.Series([cur_recalls.task_condition[0] for x in range(len(crp.index))],
                                                      index=crp.index)
                    crp['prec'] = pd.Series([prec for x in range(len(crp.index))],
                                            index=crp.index)  # todo: this needlessly makes a copy of prec for each lag... this seems like a xarray issue
                    crp['all_tf_z'] = pd.Series([all_tf_z for x in range(len(crp.index))],
                                              index=crp.index)
                    crp["aware_check"] = pd.Series([aware_check for x in range(len(crp.index))],
                                              index=crp.index)
                    all_crps = pd.concat([all_crps, crp])

            # change conditions from numerical to verbal labelsls
            recalls.loc[recalls.task_condition == 0, 'task_condition'] = "Shoebox"
            recalls.loc[recalls.task_condition == 1, 'task_condition'] = "Movie"
            recalls.loc[recalls.task_condition == 2, 'task_condition'] = "Relational"
            recalls.loc[recalls.task_condition == 3, 'task_condition'] = "Scenario"
            recalls.loc[recalls.task_condition == 4, 'task_condition'] = "Animacy"
            recalls.loc[recalls.task_condition == 5, 'task_condition'] = "Weight"
            recalls.loc[recalls.task_condition == 6, 'task_condition'] = "Front Door"
            recalls.loc[recalls.task_condition == 7, 'task_condition'] = "Constant Size"
            recalls.loc[recalls.task_condition == 8, 'task_condition'] = "Varying Size"
            recalls.loc[recalls.recall_instruction_condition == 0, 'recall_instruction_condition'] = "Free"
            recalls.loc[recalls.recall_instruction_condition == 1, 'recall_instruction_condition'] = "Serial"
            recalls.loc[recalls.instruction_condition == 0, 'instruction_condition'] = "Explicit"
            recalls.loc[recalls.instruction_condition == 1, 'instruction_condition'] = "Incidental"

            # save data used in the anals
            e1 = recalls.task_condition == "Shoebox"
            e2 = recalls.task_condition == "Front Door"
            e3 = np.logical_and(np.in1d(recalls.task_condition, ['Constant Size', "Varying Size", "Weight", "Animacy", "Scenario", "Movie", "Relational"]),
                                recalls.instruction_condition == "Incidental")
            used_data = np.logical_and(np.logical_or(e1, np.logical_or(e2, e3)), recalls.list == 0)  # in a used condition and the first list
            recalls.loc[np.logical_and(np.in1d(recalls.subject, included_subjects), used_data)].to_csv(save_name + 'Heal16implicit_data.csv')
            all_crps.to_pickle(save_name + "all_crps.pkl")

            print "Data Loaded!"
            return all_crps


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


def sample_size_table(all_crps, results_dir):

    # take mean within subject, list, and lag to give each row an unique index (currently the inxexing resets for each
    # subject, which makes it impossible to do crosstabs
    all_crps = all_crps.groupby(['subject', 'lag', 'list'], as_index=False).mean()

    # change conditions from numerical to verbal labelsls
    all_crps.loc[all_crps.task_condition == 0, 'task_condition'] = "Shoebox"
    all_crps.loc[all_crps.task_condition == 1, 'task_condition'] = "Movie"
    all_crps.loc[all_crps.task_condition == 2, 'task_condition'] = "Relational"
    all_crps.loc[all_crps.task_condition == 3, 'task_condition'] = "Scenario"
    all_crps.loc[all_crps.task_condition == 4, 'task_condition'] = "Animacy"
    all_crps.loc[all_crps.task_condition == 5, 'task_condition'] = "Weight"
    all_crps.loc[all_crps.task_condition == 6, 'task_condition'] = "Front Door"
    # all_crps.loc[all_crps.task_condition == 7, 'task_condition'] = "Constant Size"
    # all_crps.loc[all_crps.task_condition == 8, 'task_condition'] = "Varying Size"
    # all_crps.loc[all_crps.recall_instruction_condition.isnull(), 'recall_instruction_condition'] = 'Free'
    # all_crps.loc[all_crps.recall_instruction_condition == 0, 'recall_instruction_condition'] = "Free"
    # all_crps.loc[all_crps.recall_instruction_condition == 1, 'recall_instruction_condition'] = "Serial"
    all_crps.loc[all_crps.instruction_condition == 0, 'instruction_condition'] = "Explicit"
    all_crps.loc[all_crps.instruction_condition == 1, 'instruction_condition'] = "Incidental"

    # get overall sample sizes
    n_tested = pd.crosstab(all_crps[np.logical_and(all_crps.lag == 0, all_crps.list == 0)].lag,
                           [all_crps.instruction_condition, all_crps.task_condition])

    # get number excluded for being aware
    all_crps['aware_keep'] = all_crps.aware_check == 0
    n_aware = pd.crosstab(all_crps[np.logical_and(all_crps.lag == 0, all_crps.list == 0)].aware_check,
                          [all_crps.instruction_condition, all_crps.task_condition])

    # get number excluded for poor recall
    all_crps['prec_keep'] = all_crps.prec > 0
    n_Prec = pd.crosstab(all_crps[np.logical_and(all_crps.lag == 0, all_crps.list == 0)].prec_keep,
                         [all_crps.instruction_condition, all_crps.task_condition])

    # exclude the bad people
    data_filter = np.logical_and(all_crps.aware_keep, all_crps.prec_keep)
    all_crps = all_crps.loc[data_filter, :]

    # final analysed sample size
    n_included = pd.crosstab(all_crps[np.logical_and(all_crps.lag == 0, all_crps.list == 0)].lag,
                             [all_crps.instruction_condition, all_crps.task_condition])

    # compute average prec
    prec_table = all_crps.loc[np.logical_and(all_crps.lag == 0, all_crps.list == 0), :]['prec'].groupby(
        [all_crps.instruction_condition, all_crps.task_condition]).describe()

    with open(results_dir + "table_values.tex", "w") as text_file:
        text_file.write('\\newcommand\\shoeExplicit{%s}\n' % n_tested['Explicit']['Shoebox']['Free'][0])
        text_file.write('\\newcommand\\shoeIncidental{%s}\n' % n_tested['Incidental']['Shoebox']['Free'][0])
        text_file.write('\\newcommand\\doorExplicit{%s}\n' % n_tested['Explicit']['Front Door']['Free'][0])
        text_file.write('\\newcommand\\doorIncidental{%s}\n' % n_tested['Incidental']['Front Door']['Free'][0])
        text_file.write('\\newcommand\\Movie{%s}\n' % n_tested['Incidental']['Movie']['Free'][0])
        text_file.write('\\newcommand\\Relational{%s}\n' % n_tested['Incidental']['Relational']['Free'][0])
        text_file.write('\\newcommand\\Scenario{%s}\n' % n_tested['Incidental']['Scenario']['Free'][0])
        text_file.write('\\newcommand\\Animacy{%s}\n' % n_tested['Incidental']['Animacy']['Free'][0])
        text_file.write('\\newcommand\\Weight{%s}\n' % n_tested['Incidental']['Weight']['Free'][0])
        # text_file.write('\\newcommand\\ConstantFree{%s}\n' % n_tested['Incidental']['Constant Size']['Free'][0])
        # text_file.write('\\newcommand\\ConstantSerial{%s}\n' % n_tested['Incidental']['Varying Size']['Serial'][0])


        text_file.write('\\newcommand\\shoeExplicitAware{--}\n' % n_aware['Explicit']['Shoebox'][1])
        text_file.write('\\newcommand\\shoeIncidentalAware{%s}\n' % n_aware['Incidental']['Shoebox'][1])
        text_file.write('\\newcommand\\doorExplicitAware{--}\n' % n_aware['Explicit']['Front Door'][1])
        text_file.write('\\newcommand\\doorIncidentalAware{%s}\n' % n_aware['Incidental']['Front Door'][1])
        text_file.write('\\newcommand\\MovieAware{%s}\n' % n_aware['Incidental']['Movie'][1])
        text_file.write('\\newcommand\\RelationalAware{%s}\n' % n_aware['Incidental']['Relational'][1])
        text_file.write('\\newcommand\\ScenarioAware{%s}\n' % n_aware['Incidental']['Scenario'][1])
        text_file.write('\\newcommand\\AnimacyAware{%s}\n' % n_aware['Incidental']['Animacy'][1])
        text_file.write('\\newcommand\\WeightAware{%s}\n' % n_aware['Incidental']['Weight'][1])

        # # commented out because there are no people with zero prec and the 1st entry thus does not exist and throws an error
        text_file.write('\\newcommand\\shoeExplicitZeroPrec{%s}\n' % n_Prec['Explicit']['Shoebox'][1])
        text_file.write('\\newcommand\\shoeIncidentalZeroPrec{%s}\n' % n_Prec['Incidental']['Shoebox'][1])
        text_file.write('\\newcommand\\doorExplicitZeroPrec{%s}\n' % n_Prec['Explicit']['Front Door'][1])
        text_file.write('\\newcommand\\doorIncidentalZeroPrec{%s}\n' % n_Prec['Incidental']['Front Door'][1])
        text_file.write('\\newcommand\\MovieZeroPrec{%s}\n' % n_Prec['Incidental']['Movie'][1])
        text_file.write('\\newcommand\\RelationalZeroPrec{%s}\n' % n_Prec['Incidental']['Relational'][1])
        text_file.write('\\newcommand\\ScenarioZeroPrec{%s}\n' % n_Prec['Incidental']['Scenario'][1])
        text_file.write('\\newcommand\\AnimacyZeroPrec{%s}\n' % n_Prec['Incidental']['Animacy'][1])
        text_file.write('\\newcommand\\WeightZeroPrec{%s}\n' % n_Prec['Incidental']['Weight'][1])

        text_file.write('\\newcommand\\shoeExplicitIncluded{%s}\n' % n_included['Explicit']['Shoebox'][0])
        text_file.write('\\newcommand\\shoeIncidentalIncluded{%s}\n' % n_included['Incidental']['Shoebox'][0])
        text_file.write('\\newcommand\\doorExplicitIncluded{%s}\n' % n_included['Explicit']['Front Door'][0])
        text_file.write('\\newcommand\\doorIncidentalIncluded{%s}\n' % n_included['Incidental']['Front Door'][0])
        text_file.write('\\newcommand\\MovieIncluded{%s}\n' % n_included['Incidental']['Movie'][0])
        text_file.write('\\newcommand\\RelationalIncluded{%s}\n' % n_included['Incidental']['Relational'][0])
        text_file.write('\\newcommand\\ScenarioIncluded{%s}\n' % n_included['Incidental']['Scenario'][0])
        text_file.write('\\newcommand\\AnimacyIncluded{%s}\n' % n_included['Incidental']['Animacy'][0])
        text_file.write('\\newcommand\\WeightIncluded{%s}\n' % n_included['Incidental']['Weight'][0])

        text_file.write(
            '\\newcommand\\shoeExplicitPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Explicit"]["Shoebox"]["mean"],
                                                                         prec_table["Explicit"]["Shoebox"]["std"]))
        text_file.write(
            '\\newcommand\\shoeIncidentalPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Shoebox"]["mean"],
                                                                           prec_table["Incidental"]["Shoebox"]["std"]))
        text_file.write(
            '\\newcommand\\doorExplicitPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Explicit"]["Front Door"]["mean"],
                                                                         prec_table["Explicit"]["Front Door"]["std"]))
        text_file.write('\\newcommand\\doorIncidentalPrec{{{:.2f} ({:.2f})}}\n'.format(
            prec_table["Incidental"]["Front Door"]["mean"], prec_table["Incidental"]["Front Door"]["std"]))

        text_file.write(
            '\\newcommand\\MoviePrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Movie"]["mean"],
                                                                  prec_table["Incidental"]["Movie"]["std"]))
        text_file.write(
            '\\newcommand\\RelationalPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Relational"]["mean"],
                                                                       prec_table["Incidental"]["Relational"]["std"]))
        text_file.write(
            '\\newcommand\\ScenarioPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Scenario"]["mean"],
                                                                     prec_table["Incidental"]["Scenario"]["std"]))
        text_file.write(
            '\\newcommand\\AnimacyPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Animacy"]["mean"],
                                                                    prec_table["Incidental"]["Animacy"]["std"]))
        text_file.write(
            '\\newcommand\\WeightPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Weight"]["mean"],
                                                                   prec_table["Incidental"]["Weight"]["std"]))
    return all_crps


def E4_sample_size_table(all_crps, results_dir):

    # take mean within subject, list, and lag to give each row an unique index (currently the inxexing resets for each
    # subject, which makes it impossible to do crosstabs
    all_crps = all_crps.groupby(['subject', 'lag', 'list'], as_index=False).mean()

    # change conditions from numerical to verbal labelsls
    all_crps.loc[all_crps.task_condition == 0, 'task_condition'] = "Shoebox"
    all_crps.loc[all_crps.task_condition == 1, 'task_condition'] = "Movie"
    all_crps.loc[all_crps.task_condition == 2, 'task_condition'] = "Relational"
    all_crps.loc[all_crps.task_condition == 3, 'task_condition'] = "Scenario"
    all_crps.loc[all_crps.task_condition == 4, 'task_condition'] = "Animacy"
    all_crps.loc[all_crps.task_condition == 5, 'task_condition'] = "Weight"
    all_crps.loc[all_crps.task_condition == 6, 'task_condition'] = "Front Door"
    all_crps.loc[all_crps.task_condition == 7, 'task_condition'] = "Constant Size"
    all_crps.loc[all_crps.task_condition == 8, 'task_condition'] = "Varying Size"
    all_crps.loc[all_crps.recall_instruction_condition.isnull(), 'recall_instruction_condition'] = 'Free'
    all_crps.loc[all_crps.recall_instruction_condition == 0, 'recall_instruction_condition'] = "Free"
    all_crps.loc[all_crps.recall_instruction_condition == 1, 'recall_instruction_condition'] = "Serial"
    all_crps.loc[all_crps.instruction_condition == 0, 'instruction_condition'] = "Explicit"
    all_crps.loc[all_crps.instruction_condition == 1, 'instruction_condition'] = "Incidental"

    # get overall sample sizes
    n_tested = pd.crosstab(all_crps[np.logical_and(all_crps.lag == 0, all_crps.list == 0)].lag,
                           [all_crps.instruction_condition, all_crps.task_condition, all_crps.recall_instruction_condition])

    # get number excluded for being aware
    all_crps['aware_keep'] = all_crps.aware_check == 0
    n_aware = pd.crosstab(all_crps[np.logical_and(all_crps.lag == 0, all_crps.list == 0)].aware_check,
                          [all_crps.instruction_condition, all_crps.task_condition, all_crps.recall_instruction_condition])

    # get number excluded for poor recall
    all_crps['prec_keep'] = all_crps.prec > 0
    n_Prec = pd.crosstab(all_crps[np.logical_and(all_crps.lag == 0, all_crps.list == 0)].prec_keep,
                         [all_crps.instruction_condition, all_crps.task_condition, all_crps.recall_instruction_condition])

    # exclude the bad people
    data_filter = np.logical_and(all_crps.aware_keep, all_crps.prec_keep)
    all_crps = all_crps.loc[data_filter, :]

    # final analysed sample size
    n_included = pd.crosstab(all_crps[np.logical_and(all_crps.lag == 0, all_crps.list == 0)].lag,
                             [all_crps.instruction_condition, all_crps.task_condition, all_crps.recall_instruction_condition])

    # compute average prec
    prec_table = all_crps.loc[np.logical_and(all_crps.lag == 0, all_crps.list == 0), :]['prec'].groupby(
        [all_crps.instruction_condition, all_crps.task_condition, all_crps.recall_instruction_condition]).describe()

    with open(results_dir + "E4_table_values.tex", "w") as text_file:
        text_file.write('\\newcommand\\ConstantFree{%s}\n' % n_tested['Incidental']['Constant Size']['Free'][0])
        text_file.write('\\newcommand\\ConstantSerial{%s}\n' % n_tested['Incidental']['Constant Size']['Serial'][0])
        text_file.write('\\newcommand\\VaryingFree{%s}\n' % n_tested['Incidental']['Varying Size']['Free'][0])
        text_file.write('\\newcommand\\VaryingSerial{%s}\n' % n_tested['Incidental']['Varying Size']['Serial'][0])

        text_file.write('\\newcommand\\ConstantFreeAware{%s}\n' % n_aware['Incidental']['Constant Size']['Free'][1])
        text_file.write('\\newcommand\\ConstantSerialAware{%s}\n' % n_aware['Incidental']['Constant Size']['Serial'][1])
        text_file.write('\\newcommand\\VaryingFreeAware{%s}\n' % n_aware['Incidental']['Varying Size']['Free'][1])
        text_file.write('\\newcommand\\VaryingSerialAware{%s}\n' % n_aware['Incidental']['Varying Size']['Serial'][1])

        text_file.write('\\newcommand\\ConstantFreeIncluded{%s}\n' % n_included['Incidental']['Constant Size']['Free'][0])
        text_file.write('\\newcommand\\ConstantSerialIncluded{%s}\n' % n_included['Incidental']['Constant Size']['Serial'][0])
        text_file.write('\\newcommand\\VaryingFreeIncluded{%s}\n' % n_included['Incidental']['Varying Size']['Free'][0])
        text_file.write('\\newcommand\\VaryingSerialIncluded{%s}\n' % n_included['Incidental']['Varying Size']['Serial'][0])






        # text_file.write('\\newcommand\\shoeExplicitAware{--}\n' % n_aware['Explicit']['Shoebox'][1])
        # text_file.write('\\newcommand\\shoeIncidentalAware{%s}\n' % n_aware['Incidental']['Shoebox'][1])
        # text_file.write('\\newcommand\\doorExplicitAware{--}\n' % n_aware['Explicit']['Front Door'][1])
        # text_file.write('\\newcommand\\doorIncidentalAware{%s}\n' % n_aware['Incidental']['Front Door'][1])
        # text_file.write('\\newcommand\\MovieAware{%s}\n' % n_aware['Incidental']['Movie'][1])
        # text_file.write('\\newcommand\\RelationalAware{%s}\n' % n_aware['Incidental']['Relational'][1])
        # text_file.write('\\newcommand\\ScenarioAware{%s}\n' % n_aware['Incidental']['Scenario'][1])
        # text_file.write('\\newcommand\\AnimacyAware{%s}\n' % n_aware['Incidental']['Animacy'][1])
        # text_file.write('\\newcommand\\WeightAware{%s}\n' % n_aware['Incidental']['Weight'][1])
        #
        # # # commented out because there are no people with zero prec and the 1st entry thus does not exist and throws an error
        # text_file.write('\\newcommand\\shoeExplicitZeroPrec{%s}\n' % n_Prec['Explicit']['Shoebox'][1])
        # text_file.write('\\newcommand\\shoeIncidentalZeroPrec{%s}\n' % n_Prec['Incidental']['Shoebox'][1])
        # text_file.write('\\newcommand\\doorExplicitZeroPrec{%s}\n' % n_Prec['Explicit']['Front Door'][1])
        # text_file.write('\\newcommand\\doorIncidentalZeroPrec{%s}\n' % n_Prec['Incidental']['Front Door'][1])
        # text_file.write('\\newcommand\\MovieZeroPrec{%s}\n' % n_Prec['Incidental']['Movie'][1])
        # text_file.write('\\newcommand\\RelationalZeroPrec{%s}\n' % n_Prec['Incidental']['Relational'][1])
        # text_file.write('\\newcommand\\ScenarioZeroPrec{%s}\n' % n_Prec['Incidental']['Scenario'][1])
        # text_file.write('\\newcommand\\AnimacyZeroPrec{%s}\n' % n_Prec['Incidental']['Animacy'][1])
        # text_file.write('\\newcommand\\WeightZeroPrec{%s}\n' % n_Prec['Incidental']['Weight'][1])
        #
        # text_file.write('\\newcommand\\shoeExplicitIncluded{%s}\n' % n_included['Explicit']['Shoebox'][0])
        # text_file.write('\\newcommand\\shoeIncidentalIncluded{%s}\n' % n_included['Incidental']['Shoebox'][0])
        # text_file.write('\\newcommand\\doorExplicitIncluded{%s}\n' % n_included['Explicit']['Front Door'][0])
        # text_file.write('\\newcommand\\doorIncidentalIncluded{%s}\n' % n_included['Incidental']['Front Door'][0])
        # text_file.write('\\newcommand\\MovieIncluded{%s}\n' % n_included['Incidental']['Movie'][0])
        # text_file.write('\\newcommand\\RelationalIncluded{%s}\n' % n_included['Incidental']['Relational'][0])
        # text_file.write('\\newcommand\\ScenarioIncluded{%s}\n' % n_included['Incidental']['Scenario'][0])
        # text_file.write('\\newcommand\\AnimacyIncluded{%s}\n' % n_included['Incidental']['Animacy'][0])
        # text_file.write('\\newcommand\\WeightIncluded{%s}\n' % n_included['Incidental']['Weight'][0])

        # text_file.write(
        #     '\\newcommand\\shoeExplicitPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Explicit"]["Shoebox"]["mean"],
        #                                                                  prec_table["Explicit"]["Shoebox"]["std"]))
        # text_file.write(
        #     '\\newcommand\\shoeIncidentalPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Shoebox"]["mean"],
        #                                                                    prec_table["Incidental"]["Shoebox"]["std"]))
        # text_file.write(
        #     '\\newcommand\\doorExplicitPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Explicit"]["Front Door"]["mean"],
        #                                                                  prec_table["Explicit"]["Front Door"]["std"]))
        # text_file.write('\\newcommand\\doorIncidentalPrec{{{:.2f} ({:.2f})}}\n'.format(
        #     prec_table["Incidental"]["Front Door"]["mean"], prec_table["Incidental"]["Front Door"]["std"]))
        #
        # text_file.write(
        #     '\\newcommand\\MoviePrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Movie"]["mean"],
        #                                                           prec_table["Incidental"]["Movie"]["std"]))
        # text_file.write(
        #     '\\newcommand\\RelationalPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Relational"]["mean"],
        #                                                                prec_table["Incidental"]["Relational"]["std"]))
        # text_file.write(
        #     '\\newcommand\\ScenarioPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Scenario"]["mean"],
        #                                                              prec_table["Incidental"]["Scenario"]["std"]))
        # text_file.write(
        #     '\\newcommand\\AnimacyPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Animacy"]["mean"],
        #                                                             prec_table["Incidental"]["Animacy"]["std"]))


        text_file.write(
            '\\newcommand\\ConstantFreePrec{{{:.2f} ({:.2f})}}\n'.format(prec_table['mean'][0],
                                                                         prec_table['std'][0]))
        text_file.write(
            '\\newcommand\\ConstantSerialPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table['mean'][1],
                                                                         prec_table['std'][1]))
        text_file.write(
            '\\newcommand\\VaryingFreePrec{{{:.2f} ({:.2f})}}\n'.format(prec_table['mean'][2],
                                                                         prec_table['std'][2]))
        text_file.write(
            '\\newcommand\\VaryingSerialPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table['mean'][3],
                                                                         prec_table['std'][3]))
    return all_crps







def which_item(recall, presented, dictionary):
    """

    :param recall: the string typed in by the subject
    :param presented: a list of words seen by this subject so far in the experiment
    :param dictionary: a list of strings we want to consider as possible intrusions
    :return:
    """

    # does the recall exactly match a word that has been presented
    seen, seen_where = self_term_search(recall.response, presented.word)
    if seen:

        # could be a PLI
        intrusion = presented.iloc[seen_where].list != recall.list
        if intrusion:
            return -1.0 #todo: what are the standard matlab codes for ELI and PLI?

        # its not a PLI, so find the serial pos
        first_item = next(item for item, listnum in enumerate(presented.list) if listnum == recall.list)
        serial_pos = seen_where - first_item + 1.0
        return serial_pos

    # does the recall exactly match a word in the dictionary
    in_dict, where_in_dict = self_term_search(recall.response, dictionary)
    if in_dict:
        return -999.0 #todo: what are the standard matlab codes for ELI and PLI?

    # is the response a nan?
    if not type(recall.response) == str:
        if type(recall.response) ==  unicode:
            recall.response = str(recall.response)
        elif np.isnan(recall.response):
            return -1999.0 #todo: what are the standard matlab codes for ELI and PLI?

    # does the string include non letters?
    non_letter = not recall.response.isalpha()
    if non_letter:
        return -2999.0  # todo: what are the standard matlab codes for ELI and PLI?

    # the closest match based on edit distance
    recall = correct_spelling(recall, presented, dictionary)
    return which_item(recall, presented, dictionary)


def self_term_search(find_this, in_this):
    for index, word in enumerate(in_this):
        if word == find_this:
            return True, index
    return False, None


def correct_spelling(recall, presented, dictionary):

    # edit distance to each item in the pool and dictionary
    dist_to_pool = damerau_levenshtein_distance_ndarray(recall.response, np.array(presented.word))
    dist_to_dict = damerau_levenshtein_distance_ndarray(recall.response, np.array(dictionary))

    # position in distribution of dist_to_dict
    ptile = np.true_divide(sum(dist_to_dict <= np.amin(dist_to_pool)), dist_to_dict.size)

    # decide if it is a word in the pool or an ELI
    if ptile <= .1: #todo: make this a param
        corrected_recall = presented.iloc[np.argmin(dist_to_pool)].word
    else:
        corrected_recall = dictionary[np.argmin(dist_to_dict)]
    recall.response = corrected_recall
    return recall


def E4_fig(data_to_use, which_list, save_name):
    colors = ["#000000", "#808080"]
    sns.set_palette(colors)


    # setup the grid
    fig2 = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(1, 2)
    crp_axis = fig2.add_subplot(gs[0, 0])
    tf_axis = fig2.add_subplot(gs[0, 1])

    # plot crps
    data_filter = data_to_use.list == which_list
    g = sns.factorplot(x="lag", y="crp", hue="recall_instruction_condition", data=data_to_use.loc[data_filter, :],
                   hue_order=["Free", "Serial"], dodge=.25, units='subject', ax=crp_axis)
    crp_axis.set(xlabel="Lag", ylabel="Cond. Resp. Prob.", ylim=[0., .2], xticks=range(0, 11, 2),
                 xticklabels=range(-5, 6, 2))
    crp_axis.legend(title='Recall Instructions', ncol=2, labelspacing=.2, handlelength=.01, loc=2)
    plt.figure(fig2.number)
    sns.despine()
    crp_axis.annotate('A.', xy=(-.175, 1), xycoords='axes fraction')

    # plot temp factors
    data_filter = np.logical_and(data_to_use.list == which_list, data_to_use.lag == 0)
    g = sns.barplot(x="recall_instruction_condition", y=tf_col, data=data_to_use.loc[data_filter, :], order=["Free", "Serial"], ax=tf_axis) #
    tf_axis.set(xlabel="Recal Instructions", ylabel="Z(TCE)", ylim=tf_lims)
    plt.axhline(linewidth=3, linestyle='--', color='k')
    plt.figure(fig2.number)
    sns.despine()
    tf_axis.annotate('B.', xy=(-.175, 1), xycoords='axes fraction')

    fig2.savefig(save_name + '.pdf', bbox_inches='tight')
    plt.close(fig2)


def get_spc(n_perms, remake_data_file, recalls_file, save_name):

    if os.path.isfile(save_name + "all_spc.pkl") and not remake_data_file:
        all_spcs = pickle.load(open(save_name + "all_spc.pkl", "rb"))
        return all_spcs
    else:
        num_cores = multiprocessing.cpu_count() / 2
        with Parallel(n_jobs=num_cores, verbose=0) as POOL:
            # os.system('scp cbcc.psy.msu.edu:~/code/experiments/Heal16implicit/HealEtal16implicit.data.pkl \'/Users/khealey/Library/Mobile Documents/com~apple~CloudDocs/lab/code/experiments/Heal16implicit\'')
            recalls = pickle.load(open(
                recalls_file,
                "rb"))

            # loop over subjects and lists, for each isolate their data
            subjects = recalls.subject.unique()
            included_subjects = []
            n_subs = subjects.shape[0]
            si = 0.  # for a progress counter
            lists = [0]  #  doing only the first list. to do all lists change to: recalls.list.unique()
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
                    incidental = cur_recalls.instruction_condition[0] == 1
                    if aware.any() and incidental:
                        aware_check = 1
                    else:
                        aware_check = 0

                    # # compute overall recall
                    rec_mat = cur_recalls.as_matrix(range(recalls.shape[1] - 2))  # format recalls matrix for use with rdf functions
                    # prec = rdf.prec(listlen=16, recalls=rec_mat)
                    # if prec <= 0.:
                    #     continue
                    # included_subjects.append(s)
                    # # continue
                    # 
                    # # compute random temporal factor
                    # tempf_z = rdf.relative_to_random(listlen=16, recalls=rec_mat, filter_ind=None, statistic_func=rdf.tem_fact,
                    #                            data_col="tf", n_perms=n_perms, POOL=POOL)
                    # tempf_z = pd.DataFrame.from_records(tempf_z)
                    # all_tf_z = tempf_z.tf.mean()

                    # compute crp
                    spc = rdf.spc(listlen=16, recalls=rec_mat, filter_ind=None, allow_repeats=False, exclude_op=0)
                    spc = pd.DataFrame.from_records(spc)

                    # fanilize crp data: add list number, and condition ids
                    spc['subject'] = pd.Series([s for x in range(len(spc.index))], index=spc.index)
                    spc['list'] = pd.Series([l for x in range(len(spc.index))], index=spc.index)
                    spc['instruction_condition'] = pd.Series([cur_recalls.instruction_condition[0] for x in range(len(spc.index))],
                                                             index=spc.index)
                    spc['recall_instruction_condition'] = pd.Series([cur_recalls.recall_instruction_condition[0] for x in range(len(spc.index))],
                                                             index=spc.index)
                    spc['task_condition'] = pd.Series([cur_recalls.task_condition[0] for x in range(len(spc.index))],
                                                      index=spc.index)
                    # spc['prec'] = pd.Series([prec for x in range(len(spc.index))],
                    #                         index=spc.index)  # todo: this needlessly makes a copy of prec for each lag... this seems like a xarray issue
                    # spc['all_tf_z'] = pd.Series([all_tf_z for x in range(len(spc.index))],
                    #                           index=spc.index)
                    spc["aware_check"] = pd.Series([aware_check for x in range(len(spc.index))],
                                              index=spc.index)
                    all_spcs = pd.concat([all_spcs, spc])

            # change conditions from numerical to verbal labelsls
            recalls.loc[recalls.task_condition == 0, 'task_condition'] = "Shoebox"
            recalls.loc[recalls.task_condition == 1, 'task_condition'] = "Movie"
            recalls.loc[recalls.task_condition == 2, 'task_condition'] = "Relational"
            recalls.loc[recalls.task_condition == 3, 'task_condition'] = "Scenario"
            recalls.loc[recalls.task_condition == 4, 'task_condition'] = "Animacy"
            recalls.loc[recalls.task_condition == 5, 'task_condition'] = "Weight"
            recalls.loc[recalls.task_condition == 6, 'task_condition'] = "Front Door"
            recalls.loc[recalls.task_condition == 7, 'task_condition'] = "Constant Size"
            recalls.loc[recalls.task_condition == 8, 'task_condition'] = "Varying Size"
            recalls.loc[recalls.recall_instruction_condition == 0, 'recall_instruction_condition'] = "Free"
            recalls.loc[recalls.recall_instruction_condition == 1, 'recall_instruction_condition'] = "Serial"
            recalls.loc[recalls.instruction_condition == 0, 'instruction_condition'] = "Explicit"
            recalls.loc[recalls.instruction_condition == 1, 'instruction_condition'] = "Incidental"

            # save data used in the anals
            e1 = recalls.task_condition == "Shoebox"
            e2 = recalls.task_condition == "Front Door"
            e3 = np.logical_and(np.in1d(recalls.task_condition, ["Weight", "Animacy", "Scenario", "Movie", "Relational"]),
                                recalls.instruction_condition == "Incidental")
            used_data = np.logical_and(np.logical_or(e1, np.logical_or(e2, e3)), recalls.list == 0)  # in a used condition and the first list
            recalls.loc[np.logical_and(np.in1d(recalls.subject, included_subjects), used_data)].to_csv(save_name + 'Heal16implicit_data.csv')
            all_spcs.to_pickle(save_name + "all_spc.pkl")

            print "Data Loaded!"
            return all_spcs