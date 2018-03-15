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
from pyxdameraulevenshtein import damerau_levenshtein_distance_ndarray
import xarray as xr
from matplotlib import rcParams
from cbcc_tools.beh_anal import recall_dynamics as cbcc
from cycler import cycler
from scipy import stats
import tcm_Heal16implicit as tcm
from scipy.optimize import differential_evolution
from scipy.stats import ttest_ind


# figure style params
sns.set_style("ticks")
sns.set_context("talk", font_scale=2.5, rc={"lines.linewidth": 4})
colors = ["#000000", "#808080", "#D3D3D3"]  # black and white
sns.set_palette(colors)

# params for temporal factor plots
tf_lims = [-.025, .2]
tf_col ='all_tf_z'

# figure size
one_col = 3.5
two_col = one_col*2
base_height = 2.5


def make_psiturk_recall_matrix(remake_data_file, dict_path, save_file):
    if os.path.isfile(save_file + ".pkl") and not remake_data_file:
        recalls = pickle.load(open(save_file + ".pkl", "rb"))
        return recalls

    # load all the data for all experiments from the file made from the master database on cbcc
    data = pickle.load(open("HealEtal16implicit.data.raw.pkl", "rb"))

    # load the E1--3 data from the cvs file created for the first submission
    sub1_data = pd.DataFrame.from_csv('/Users/khealey/code/experiments/Heal16implicit/dissemination/'
                                      'manuscript/jml/first_submission/figures/Heal16implicit_data.csv')

    # load or create the recalls matrix
    sub1_ss_list = sub1_data.subject.unique()

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

        # # do only a few subjects to speedup debugging
        # if loop > 50:
        #     break

        print(loop/n_ss*100)
        s_filter = data.uniqueid == s
        recalls_filter = data.phase == 'recall'
        study_filter = data.phase == 'study'
        awareness_filter = data.aware_question == 'awarenesscheck'
        aware = data.loc[s_filter & awareness_filter, 'aware_ans']
        cur_recalls = data.loc[s_filter & recalls_filter, ['list', 'response', 'instruction_condition','task_condition', 'recall_instruction_condition']]
        cur_items = data.loc[s_filter & study_filter, ['list', 'word']]

        gender_filter = data.aware_question == 'gender'
        gender = data.loc[s_filter & gender_filter, 'aware_ans'].values

        english_filter = data.aware_question == 'english'
        english = data.loc[s_filter & english_filter, 'aware_ans'].values

        edu_filter = data.aware_question == 'edu'
        edu = data.loc[s_filter & edu_filter, 'aware_ans'].values

        age_filter = data.strategy == 'age'
        age = data.loc[s_filter & age_filter, 'strategy_description'].values

        # we are only interested in people who were included in sub1 or are in the new E4
        in_e4 = ~cur_recalls.recall_instruction_condition.isnull().all()
        in_sub1 = s in sub1_ss_list
        if not in_e4 and not in_sub1:
            # print ('excluded subject!!')
            continue

        # somehow, there seems to be some uniqueid's that are have two of each list.... just move on if that is the case
        if cur_items.shape[0] != 32 and cur_items.shape[0] != 16:
            # print ("DUP!")
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
            # recall = recalled_this_list.tail(1)


            # we need to add the subject id and conditions to the beginning of the line
            sp.extend((s, recall.list, recall.instruction_condition, recall.task_condition, recall.recall_instruction_condition, aware, gender, english, edu, age)) #sp.insert(0, s)
            op.extend(('subject', 'list', 'instruction_condition', 'task_condition', 'recall_instruction_condition', 'aware', "gender", "english", "edu", "age")) #op.insert(0, 'subject')
            recalls = recalls.append(pd.DataFrame([sp], columns=tuple(op)))

    recalls.set_index('subject')
    recalls.to_pickle(save_file + ".pkl")
    return recalls


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


def load_the_data(n_perms, remake_data_file, recalls_file, save_name):

    if os.path.isfile(save_name + "all_crps.pkl") and not remake_data_file:
        all_crps = pickle.load(open(save_name + "all_crps.pkl", "rb"))
        return all_crps
    else:
        num_cores = multiprocessing.cpu_count() / 2
        with Parallel(n_jobs=num_cores, verbose=0) as POOL:
            recalls = pickle.load(open(
                recalls_file,
                "rb"))

            # loop over subjects and lists, for each isolate their data
            subjects = recalls.subject.unique()
            included_subjects = []
            n_subs = subjects.shape[0]
            si = 0.  # for a progress counter
            lists = recalls.list.unique()
            all_crps = pd.DataFrame()
            for s in subjects:
                si += 1
                print (si / n_subs * 100.)
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
                    output_cols = range(recalls.columns.get_loc('age'))
                    rec_mat = cur_recalls.as_matrix(output_cols)  # format recalls matrix for use with rdf functions
                    prec = rdf.prec(listlen=16, recalls=rec_mat)
                    if prec <= 0.:
                        continue
                    included_subjects.append(s)

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
            recalls.loc[recalls.recall_instruction_condition.isnull(), 'recall_instruction_condition'] = 'Free'
            recalls.loc[recalls.recall_instruction_condition == 0, 'recall_instruction_condition'] = "Free"
            recalls.loc[recalls.recall_instruction_condition == 1, 'recall_instruction_condition'] = "Serial"
            recalls.loc[recalls.instruction_condition == 0, 'instruction_condition'] = "Explicit"
            recalls.loc[recalls.instruction_condition == 1, 'instruction_condition'] = "Incidental"

            # save data used in the anals
            e1 = recalls.task_condition == "Shoebox"
            e2 = recalls.task_condition == "Front Door"
            e3and4 = np.logical_and(np.in1d(recalls.task_condition,
                                            ['Constant Size', "Varying Size", "Weight", "Animacy",
                                             "Scenario", "Movie", "Relational"]),
                                    recalls.instruction_condition == "Incidental")
            used_data = np.logical_or(e1, np.logical_or(e2, e3and4)) # in a used condition and the first list
            recalls.loc[np.logical_and(np.in1d(recalls.subject, included_subjects), used_data)].to_csv(save_name + 'Heal16implicit_data.csv')
            all_crps.to_pickle(save_name + "all_crps.pkl")

            print ("Data Loaded!")
            return all_crps


def make_xarray(data, list_number):

    # get sample sizes (aware and not) then get rid of aware people in the incidential conditons
    data['aware'] = data['aware'].str.contains('yes')
    sample_sizes_aware_counts = pd.crosstab(data.aware, [data.instruction_condition, data.task_condition])
    data = data.loc[np.logical_or(~data['aware'], data['instruction_condition'] == 'Explicit'), :]
    data = data.loc[data.list == list_number]
    sample_sizes_included_counts = pd.crosstab(data.aware, [data.instruction_condition, data.task_condition])

    # condition vector
    coords = {
        'subject': data.subject,
    }
    instruction_condition = xr.DataArray(data.instruction_condition, dims=('subject'), coords=coords)
    task_condition = xr.DataArray(data.task_condition, dims=('subject'), coords=coords)
    recall_instruction_condition = xr.DataArray(data.recall_instruction_condition, dims=('subject'), coords=coords)
    n_outputs = 24
    coords.update({'output_position': range(n_outputs+1)})
    coords.update({'list': [list_number]})
    rec_mat = data[data.columns[0:n_outputs+1]].values[:, np.newaxis, :]  # note adding an new axis for list

    # translate from beh_toolbox coding to cbcc_tools coding
    rec_mat = rec_mat - 1  # convert to zero indexing
    rec_mat[rec_mat == -1-1] = -2  # PLIs
    rec_mat[rec_mat == -999.0-1] = -3  # PLIs
    rec_mat[rec_mat == -1999.0-1] = -4  # nans in middle of sequence
    rec_mat[rec_mat == -2999.0-1] = -4  # non alpha string
    rec_mat[np.isnan(rec_mat)] = -1  # post last recall

    rec_mat = rec_mat.astype('int')
    recalls = xr.DataArray(rec_mat,
                           dims=('subject', 'list', 'output_position'), coords=coords)
    ds = xr.Dataset({
        'recalls': recalls,
        'instruction_condition': instruction_condition,
        'task_condition': task_condition,
        'recall_instruction_condition': recall_instruction_condition
    })
    ds.attrs['n_items_per_list'] = 16
    return ds, sample_sizes_aware_counts, sample_sizes_included_counts


def sample_size_table(all_crps, results_dir, recalls):

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
        text_file.write('\\newcommand\\ConstantFree{%s}\n' % n_tested['Incidental']['Constant Size']['Free'][0])
        text_file.write('\\newcommand\\ConstantSerial{%s}\n' % n_tested['Incidental']['Constant Size']['Serial'][0])
        text_file.write('\\newcommand\\VaryingFree{%s}\n' % n_tested['Incidental']['Varying Size']['Free'][0])
        text_file.write('\\newcommand\\VaryingSerial{%s}\n' % n_tested['Incidental']['Varying Size']['Serial'][0])

        text_file.write('\\newcommand\\shoeExplicitAware{--}\n' % n_aware['Explicit']['Shoebox']['Free'][1])
        text_file.write('\\newcommand\\shoeIncidentalAware{%s}\n' % n_aware['Incidental']['Shoebox']['Free'][1])
        text_file.write('\\newcommand\\doorExplicitAware{--}\n' % n_aware['Explicit']['Front Door']['Free'][1])
        text_file.write('\\newcommand\\doorIncidentalAware{%s}\n' % n_aware['Incidental']['Front Door']['Free'][1])
        text_file.write('\\newcommand\\MovieAware{%s}\n' % n_aware['Incidental']['Movie']['Free'][1])
        text_file.write('\\newcommand\\RelationalAware{%s}\n' % n_aware['Incidental']['Relational']['Free'][1])
        text_file.write('\\newcommand\\ScenarioAware{%s}\n' % n_aware['Incidental']['Scenario']['Free'][1])
        text_file.write('\\newcommand\\AnimacyAware{%s}\n' % n_aware['Incidental']['Animacy']['Free'][1])
        text_file.write('\\newcommand\\WeightAware{%s}\n' % n_aware['Incidental']['Weight']['Free'][1])
        text_file.write('\\newcommand\\ConstantFreeAware{%s}\n' % n_aware['Incidental']['Constant Size']['Free'][1])
        text_file.write('\\newcommand\\ConstantSerialAware{%s}\n' % n_aware['Incidental']['Constant Size']['Serial'][1])
        text_file.write('\\newcommand\\VaryingFreeAware{%s}\n' % n_aware['Incidental']['Varying Size']['Free'][1])
        text_file.write('\\newcommand\\VaryingSerialAware{%s}\n' % n_aware['Incidental']['Varying Size']['Serial'][1])

        text_file.write('\\newcommand\\shoeExplicitZeroPrec{%s}\n' % n_Prec['Explicit']['Shoebox']['Free'][1])
        text_file.write('\\newcommand\\shoeIncidentalZeroPrec{%s}\n' % n_Prec['Incidental']['Shoebox']['Free'][1])
        text_file.write('\\newcommand\\doorExplicitZeroPrec{%s}\n' % n_Prec['Explicit']['Front Door']['Free'][1])
        text_file.write('\\newcommand\\doorIncidentalZeroPrec{%s}\n' % n_Prec['Incidental']['Front Door']['Free'][1])
        text_file.write('\\newcommand\\MovieZeroPrec{%s}\n' % n_Prec['Incidental']['Movie']['Free'][1])
        text_file.write('\\newcommand\\RelationalZeroPrec{%s}\n' % n_Prec['Incidental']['Relational']['Free'][1])
        text_file.write('\\newcommand\\ScenarioZeroPrec{%s}\n' % n_Prec['Incidental']['Scenario']['Free'][1])
        text_file.write('\\newcommand\\AnimacyZeroPrec{%s}\n' % n_Prec['Incidental']['Animacy']['Free'][1])
        text_file.write('\\newcommand\\WeightZeroPrec{%s}\n' % n_Prec['Incidental']['Weight']['Free'][1])
        text_file.write('\\newcommand\\ConstantFreeZeroPrec{%s}\n' % n_Prec['Incidental']['Constant Size']['Free'][1])
        text_file.write('\\newcommand\\ConstantSerialZeroPrec{%s}\n' % n_Prec['Incidental']['Constant Size']['Serial'][1])
        text_file.write('\\newcommand\\VaryingFreeZeroPrec{%s}\n' % n_Prec['Incidental']['Varying Size']['Free'][1])
        text_file.write('\\newcommand\\VaryingSerialZeroPrec{%s}\n' % n_Prec['Incidental']['Varying Size']['Serial'][1])

        text_file.write('\\newcommand\\shoeExplicitIncluded{%s}\n' % n_included['Explicit']['Shoebox']['Free'][0])
        text_file.write('\\newcommand\\shoeIncidentalIncluded{%s}\n' % n_included['Incidental']['Shoebox']['Free'][0])
        text_file.write('\\newcommand\\doorExplicitIncluded{%s}\n' % n_included['Explicit']['Front Door']['Free'][0])
        text_file.write('\\newcommand\\doorIncidentalIncluded{%s}\n' % n_included['Incidental']['Front Door']['Free'][0])
        text_file.write('\\newcommand\\MovieIncluded{%s}\n' % n_included['Incidental']['Movie']['Free'][0])
        text_file.write('\\newcommand\\RelationalIncluded{%s}\n' % n_included['Incidental']['Relational']['Free'][0])
        text_file.write('\\newcommand\\ScenarioIncluded{%s}\n' % n_included['Incidental']['Scenario']['Free'][0])
        text_file.write('\\newcommand\\AnimacyIncluded{%s}\n' % n_included['Incidental']['Animacy']['Free'][0])
        text_file.write('\\newcommand\\WeightIncluded{%s}\n' % n_included['Incidental']['Weight']['Free'][0])
        text_file.write('\\newcommand\\ConstantFreeIncluded{%s}\n' % n_included['Incidental']['Constant Size']['Free'][0])
        text_file.write('\\newcommand\\ConstantSerialIncluded{%s}\n' % n_included['Incidental']['Constant Size']['Serial'][0])
        text_file.write('\\newcommand\\VaryingFreeIncluded{%s}\n' % n_included['Incidental']['Varying Size']['Free'][0])
        text_file.write('\\newcommand\\VaryingSerialIncluded{%s}\n' % n_included['Incidental']['Varying Size']['Serial'][0])

        text_file.write(
            '\\newcommand\\shoeExplicitPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Explicit"]["Shoebox"]['Free'],
                                                                         prec_table["std"]["Explicit"]["Shoebox"]['Free']))
        text_file.write(
            '\\newcommand\\shoeIncidentalPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Shoebox"]['Free'],
                                                                           prec_table["std"]["Incidental"]["Shoebox"]['Free']))
        text_file.write(
            '\\newcommand\\doorExplicitPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Explicit"]["Front Door"]['Free'],
                                                                         prec_table["std"]["Explicit"]["Front Door"]['Free']))
        text_file.write('\\newcommand\\doorIncidentalPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Front Door"]['Free'],
                                                                                       prec_table["std"]["Incidental"]["Front Door"]['Free']))

        text_file.write(
            '\\newcommand\\MoviePrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Movie"]['Free'],
                                                                  prec_table["std"]["Incidental"]["Movie"]['Free']))
        text_file.write(
            '\\newcommand\\RelationalPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Relational"]['Free'],
                                                                       prec_table["std"]["Incidental"]["Relational"]['Free']))
        text_file.write(
            '\\newcommand\\ScenarioPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Scenario"]['Free'],
                                                                     prec_table["std"]["Incidental"]["Scenario"]['Free']))
        text_file.write(
            '\\newcommand\\AnimacyPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Animacy"]['Free'],
                                                                    prec_table["std"]["Incidental"]["Animacy"]['Free']))
        text_file.write(
            '\\newcommand\\WeightPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Weight"]['Free'],
                                                                   prec_table["std"]["Incidental"]["Weight"]['Free']))
        text_file.write(
            '\\newcommand\\ConstantFreePrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Constant Size"]['Free'],
                                                                   prec_table["std"]["Incidental"]["Constant Size"]['Free']))
        text_file.write(
            '\\newcommand\\ConstantSerialPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Constant Size"]['Serial'],
                                                                   prec_table["std"]["Incidental"]["Constant Size"]['Serial']))
        text_file.write(
            '\\newcommand\\VaryingFreePrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Varying Size"]['Free'],
                                                                   prec_table["std"]["Incidental"]["Varying Size"]['Free']))
        text_file.write(
            '\\newcommand\\VaryingSerialPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["mean"]["Incidental"]["Varying Size"]['Serial'],
                                                                   prec_table["std"]["Incidental"]["Varying Size"]['Serial']))

        # demographics for E4
        constant = all_crps.task_condition == "Constant Size"
        varying = all_crps.task_condition == "Varying Size"
        these_data = all_crps[varying | constant]
        task_cond_filter = these_data.task_condition == "Varying Size"
        recall_cond_filter = these_data.recall_instruction_condition == "Serial"
        varying_serial = task_cond_filter & recall_cond_filter
        these_data = these_data[~varying_serial]
        age = []
        males = 0
        females = 0
        others = 0
        notans = 0
        eng_yes = 0
        eng_no = 0
        eng_skip = 0
        none = 0
        hig = 0
        ass = 0
        bac = 0
        mas = 0
        high = 0
        notsay = 0
        for s in these_data.subject.unique():
            age.append(recalls[recalls.subject == s].age[0].astype(float)[0])
            if recalls[recalls.subject == s].gender[0][0] == "male":
                males += 1
            elif recalls[recalls.subject == s].gender[0][0] == "female":
                females += 1
            elif recalls[recalls.subject == s].gender[0][0] == "other":
                others += 1
            else:
                notans += 1
            if recalls[recalls.subject == s].english[0][0] == 'Yes':
                eng_yes += 1
            elif recalls[recalls.subject == s].english[0][0] == 'No':
                eng_no += 1
            else:
                eng_skip += 1
            if recalls[recalls.subject == s].edu[0][0] == "none":
                none += 1
            elif recalls[recalls.subject == s].edu[0][0] == "hig":
                hig += 1
            elif recalls[recalls.subject == s].edu[0][0] == "ass":
                ass += 1
            elif recalls[recalls.subject == s].edu[0][0] == "bac":
                bac += 1
            elif recalls[recalls.subject == s].edu[0][0] == "mas":
                mas += 1
            elif recalls[recalls.subject == s].edu[0][0] == "high":
                high += 1
            elif recalls[recalls.subject == s].edu[0][0] == "notsay":
                notsay += 1
        total_expected = 1591
        assert males + females + others + notans == total_expected
        assert eng_yes + eng_no + eng_skip == total_expected
        assert none + hig + ass + bac + mas + high + notsay == total_expected

        # age
        text_file.write('\\newcommand\\age{{{:.2f} ($SD = {:.2f}$)}}\n'.format(np.mean(age), np.std(age)))

        # gender
        text_file.write('\\newcommand\\males{{{:d}}}\n'.format(males))
        text_file.write('\\newcommand\\females{{{:d}}}\n'.format(females))
        text_file.write('\\newcommand\\others{{{:d}}}\n'.format(others))
        text_file.write('\\newcommand\\notans{{{:d}}}\n'.format(notans))

        # english
        text_file.write('\\newcommand\\engY{{{:d}}}\n'.format(eng_yes))
        text_file.write('\\newcommand\\engN{{{:d}}}\n'.format(eng_no))
        text_file.write('\\newcommand\\engS{{{:d}}}\n'.format(eng_skip))

        # edu
        text_file.write('\\newcommand\\noed{{{:d}}}\n'.format(none))
        text_file.write('\\newcommand\\hschool{{{:d}}}\n'.format(hig))
        text_file.write('\\newcommand\\ass{{{:d}}}\n'.format(ass))
        text_file.write('\\newcommand\\bac{{{:d}}}\n'.format(bac))
        text_file.write('\\newcommand\\mas{{{:d}}}\n'.format(mas))
        text_file.write('\\newcommand\\phd{{{:d}}}\n'.format(high))
        text_file.write('\\newcommand\\notansed{{{:d}}}\n'.format(notsay))






    return all_crps


def encoding_instruct_fig(data_to_use, which_list, save_name):
    if which_list == 0:
        ylims_tf = [-.025, .2]
        ylims_crp = [0., .2]
    elif which_list == 1:
        ylims_tf = [-.025, .3]
        ylims_crp = [0., .3]
    else:
        print("WTF?")
        return

    # setup the grid
    fig2 = plt.figure(figsize=(two_col, two_col/2))
    gs = gridspec.GridSpec(1, 2)
    crp_axis = fig2.add_subplot(gs[0, 0])
    tf_axis = fig2.add_subplot(gs[0, 1])

    # plot crps
    data_filter = data_to_use.list == which_list
    rcParams['lines.linewidth'] = 1
    rcParams['lines.markersize'] = 0
    g = sns.factorplot(x="lag", y="crp", hue="instruction_condition", data=data_to_use.loc[data_filter, :],
                   hue_order=["Explicit", "Incidental"], dodge=.25, units='subject', ax=crp_axis)
    crp_axis.set(xlabel="Lag", ylabel="Cond. Resp. Prob.", ylim=ylims_crp, xticks=range(0, 11, 2),
                 yticks=np.arange(0, ylims_crp[1]+0.05, 0.05),
                 xticklabels=range(-5, 6, 2))
    crp_axis.legend(title='Encoding Instructions', ncol=2, labelspacing=.2, handlelength=.01, loc=2)
    plt.figure(fig2.number)
    sns.despine()
    crp_axis.annotate('A.', xy=(-.21, 1), xycoords='axes fraction', weight='bold')

    # plot temp factors
    data_filter = np.logical_and(data_to_use.list == which_list, data_to_use.lag == 0)
    g = sns.barplot(x="instruction_condition", y=tf_col, data=data_to_use.loc[data_filter, :],
                    order=["Explicit", "Incidental"], ax=tf_axis) #
    tf_axis.set(xlabel="Encoding Instructions", ylabel="z(TCE)", ylim=ylims_tf,
                yticks=np.arange(0, ylims_tf[1]+0.05, 0.05))
    tf_axis.lines[0].set_color('grey')
    tf_axis.lines[1].set_color('black')
    plt.axhline(linewidth=1, linestyle='--', color='k')
    plt.figure(fig2.number)
    sns.despine()
    tf_axis.annotate('B.', xy=(-.19, 1), xycoords='axes fraction', weight='bold')

    fig2.savefig(save_name + '.pdf', bbox_inches='tight')
    plt.close(fig2)


def spc_encoding_instructions_fig(to_plot, task, save_file):
    # spc/pfr for list 0
    fig = plt.figure(figsize=(one_col, base_height*2))
    gs = gridspec.GridSpec(2, 1)
    spc_axis = fig.add_subplot(gs[0, 0])
    e1_explicit_filter = np.logical_and(to_plot.instruction_condition == 'Explicit', to_plot.task_condition == task)
    e1_implicit_filter = np.logical_and(to_plot.instruction_condition == 'Incidental', to_plot.task_condition == task)
    cbcc.spc_plot(to_plot.spc[e1_explicit_filter], ax=spc_axis)
    cbcc.spc_plot(to_plot.spc[e1_implicit_filter], ax=spc_axis)
    plt.legend(['Explicit', "Incidental"])
    spc_axis.set_xlabel('')
    spc_axis.get_xaxis().set_ticklabels([])
    spc_axis.annotate('A.', xy=(-.155, 1), xycoords='axes fraction', weight='bold')
    pfr_axis = fig.add_subplot(gs[1, 0])
    cbcc.pfr_plot(to_plot.pfr[e1_explicit_filter], ax=pfr_axis)
    cbcc.pfr_plot(to_plot.pfr[e1_implicit_filter], ax=pfr_axis)
    pfr_axis.set_ylim(0, 0.5)
    pfr_axis.annotate('B.', xy=(-.155, 1), xycoords='axes fraction', weight='bold')
    plt.savefig(save_file)
    plt.close()


def e3fig(data, save_file, which_list):
    if which_list == 0:
        ylims_tf = [-.025, .2]
        ylims_crp = [0., .2]
    elif which_list == 1:
        ylims_tf = [-.025, .3]
        ylims_crp = [0., .3]
    else:
        print("WTF?")
        return

    # setup the grid
    fig1 = plt.figure(figsize=(two_col, base_height*1.5))
    ax1 = plt.subplot2grid((2, 5), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((2, 5), (0, 1), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((2, 5), (0, 2), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((2, 5), (0, 3), rowspan=1, colspan=1)
    ax5 = plt.subplot2grid((2, 5), (0, 4), rowspan=1, colspan=1)
    # ax6 = plt.subplot2grid((2, 6), (0, 5), rowspan=1, colspan=1)
    ax7 = plt.subplot2grid((2, 5), (1, 0), rowspan=1, colspan=5)

    rcParams['lines.linewidth'] = 0.6
    rcParams['lines.markersize'] = 0
    rcParams['axes.prop_cycle'] = cycler('color', ['#000000', '#000000', '#000000', '#000000', '#000000', '#000000'])

    # make the figure each axis at a time

    # AX1
    cond1_data = data[(data['task_condition'] == 'Weight')]  # just the weight condition

    g = sns.factorplot(x="lag", y="crp", data=cond1_data, units='subject', ax=ax1)
    ax1.set(ylim=ylims_crp, xticks=[0, 5, 10], xticklabels=[-5, 0, 5], yticks=np.arange(0, ylims_crp[1]+0.05, 0.05))
    ax1.xaxis.label.set_visible(False)
    ax1.set(ylabel="Cond. Resp. Prob.")
    plt.close()

    # AX2
    cond2_data = data[(data['task_condition'] == "Animacy")]  # just the animacy condition

    sns.factorplot(x="lag", y="crp", data=cond2_data, units='subject', ax=ax2, color='#000000')
    ax2.set(ylim=ylims_crp, xticks=[0, 5, 10], xticklabels=[-5, 0, 5], yticks=np.arange(0, ylims_crp[1]+0.05, 0.05))
    ax2.get_yaxis().set_ticklabels([])
    ax2.yaxis.label.set_visible(False)
    ax2.xaxis.label.set_visible(False)
    #ax2.yaxis.set_visible(False)
    plt.close()

    # AX3
    cond3_data = data[(data['task_condition'] == "Scenario")]  # just the scenario condition

    sns.factorplot(x="lag", y="crp", data=cond3_data, units='subject', ax=ax3, color='#000000')
    ax3.set(ylim=ylims_crp, xticks=[0, 5, 10], xticklabels=[-5, 0, 5], yticks=np.arange(0, ylims_crp[1]+0.05, 0.05))
    ax3.get_yaxis().set_ticklabels([])
    ax3.yaxis.label.set_visible(False)
    # ax3.xaxis.label.set_visible(False)
    #ax2.yaxis.set_visible(False)
    ax3.set(xlabel="Lag")
    plt.close()

    # AX4
    cond4_data = data[(data['task_condition'] == "Movie")]  # just the movie condition

    sns.factorplot(x="lag", y="crp", data=cond4_data, units='subject', ax=ax4, color='#000000')
    ax4.set(ylim=ylims_crp, xticks=[0, 5, 10], xticklabels=[-5, 0, 5], yticks=np.arange(0, ylims_crp[1]+0.05, 0.05))
    ax4.get_yaxis().set_ticklabels([])
    ax4.yaxis.label.set_visible(False)
    ax4.xaxis.label.set_visible(False)
    #ax2.yaxis.set_visible(False)
    plt.close()

    # AX5
    cond5_data = data[(data['task_condition'] == "Relational")]  # just the relational condition

    sns.factorplot(x="lag", y="crp", data=cond5_data, units='subject', ax=ax5, color='#000000')
    ax5.set(ylim=ylims_crp, xticks=[0, 5, 10], xticklabels=[-5, 0, 5], yticks=np.arange(0, ylims_crp[1]+0.05, 0.05))
    ax5.get_yaxis().set_ticklabels([])
    ax5.yaxis.label.set_visible(False)
    ax5.xaxis.label.set_visible(False)
    #ax2.yaxis.set_visible(False)
    plt.close()

    # AX7
    barplot_data = data
    barplot_data = barplot_data[(barplot_data['lag'] == 0)]  # only need one lag

    g = sns.barplot(x="task_condition", y="all_tf_z", data=barplot_data, ax=ax7,
                    order=["Weight", "Animacy", "Scenario", "Movie", "Relational"])
    g.set_xticklabels(["Weight", "Animacy", "Moving Scenario", "Movie", "Relational"])
    for line, l in enumerate(ax7.lines):
        ax7.lines[line].set_color('grey')
    ax7.set(xlabel="Judgment Task", ylabel="z(TCE)", ylim=ylims_tf, yticks=np.arange(0, ylims_tf[1]+0.05, 0.05))
    ax7.axhline(y=0, linewidth=1, linestyle='--', color='k')

    # save the figure
    plt.savefig(save_file + '.pdf')
    plt.close()


def e3_spc_fig(to_plot, save_file):
    # spc/pfr for list 0
    fig = plt.figure(figsize=(two_col, base_height*1.5))
    gs = gridspec.GridSpec(2, 5)
    rcParams['lines.linewidth'] = 1
    rcParams['lines.markersize'] = 1

    recall_instruction_cond_filter = to_plot.recall_instruction_condition == "Free"
    instruction_cond_filter = to_plot.instruction_condition == "Incidental"
    cond_filter = np.logical_and(recall_instruction_cond_filter, instruction_cond_filter)
    task_list = ["Weight", "Animacy", "Scenario", "Movie", "Relational"]
    task_col = 0
    for task in task_list:
        spc_axis = fig.add_subplot(gs[0, task_col])
        cbcc.spc_plot(to_plot.spc[np.logical_and(cond_filter, to_plot.task_condition == task)], ax=spc_axis)
        spc_axis.set_title(task)
        spc_axis.set_xlabel('')
        spc_axis.get_xaxis().set_ticklabels([])
        pfr_axis = fig.add_subplot(gs[1, task_col])
        cbcc.pfr_plot(to_plot.pfr[np.logical_and(cond_filter, to_plot.task_condition == task)], ax=pfr_axis)
        pfr_axis.set_ylim(0, 0.5)
        spc_axis.set_ylabel('Recall Prob.')
        pfr_axis.set_ylabel('Prob. $1^{st}$ Recall')
        if task_col > 0:
            spc_axis.set_ylabel('')
            spc_axis.get_yaxis().set_ticklabels([])
            pfr_axis.set_ylabel('')
            pfr_axis.get_yaxis().set_ticklabels([])
        task_col += 1
    plt.savefig(save_file)
    plt.close()


def e4_spc_fig(to_plot, save_file):
    fig = plt.figure(figsize=(one_col, base_height*2))  # 1 col width = 2.8
    gs = gridspec.GridSpec(2, 1)
    spc_axis = fig.add_subplot(gs[0, 0])
    constant_free_filter = np.logical_and(to_plot.recall_instruction_condition == "Free",
                                          np.logical_and(to_plot.instruction_condition == 'Incidental',
                                                         to_plot.task_condition == "Constant Size"))
    varying_free_filter = np.logical_and(to_plot.recall_instruction_condition == "Free",
                                         np.logical_and(to_plot.instruction_condition == 'Incidental',
                                                        to_plot.task_condition == "Varying Size"))

    constant_serial_filter = np.logical_and(to_plot.recall_instruction_condition == "Serial",
                                            np.logical_and(to_plot.instruction_condition == 'Incidental',
                                                           to_plot.task_condition == "Constant Size"))
    varying_serial_filter = np.logical_and(to_plot.recall_instruction_condition == "Serial",
                                           np.logical_and(to_plot.instruction_condition == 'Incidental',
                                                          to_plot.task_condition == "Varying Size"))
    cbcc.spc_plot(to_plot.spc[varying_free_filter], ax=spc_axis)
    cbcc.spc_plot(to_plot.spc[constant_free_filter], ax=spc_axis)
    # cbcc.spc_plot(to_plot.spc[varying_free_filter], ax=spc_axis, color='#000000')
    # spc_axis.lines[-1].set_marker('s')
    cbcc.spc_plot(to_plot.spc[constant_serial_filter], ax=spc_axis)
    # spc_axis.lines[-1].set_color('#808080')
    # cbcc.spc_plot(to_plot.spc[varying_serial_filter], ax=spc_axis, color='#808080')
    # spc_axis.lines[-1].set_marker('s')
    # spc_axis.lines[-1].set_color('#808080')
    plt.legend(["Varying-Free", "Constant-Free", "Constant-Serial"])
    spc_axis.set_xlabel('')
    spc_axis.get_xaxis().set_ticklabels([])
    spc_axis.annotate('A.', xy=(-.155, 1), xycoords='axes fraction', weight='bold')
    pfr_axis = fig.add_subplot(gs[1, 0])
    cbcc.pfr_plot(to_plot.pfr[varying_free_filter], ax=pfr_axis)
    cbcc.pfr_plot(to_plot.pfr[constant_free_filter], ax=pfr_axis)
    # cbcc.pfr_plot(to_plot.pfr[varying_free_filter], ax=pfr_axis, color='#000000')
    # pfr_axis.lines[-1].set_marker('s')
    # pfr_axis.lines[-1].set_color('#000000')
    cbcc.pfr_plot(to_plot.pfr[constant_serial_filter], ax=pfr_axis)
    # pfr_axis.lines[-1].set_color('#808080')
    # cbcc.pfr_plot(to_plot.pfr[varying_serial_filter], ax=pfr_axis, color='#808080')
    # pfr_axis.lines[-1].set_marker('s')
    # pfr_axis.lines[-1].set_color('#808080')
    pfr_axis.set_ylim(0, 0.5)
    pfr_axis.annotate('B.', xy=(-.155, 1), xycoords='axes fraction', weight='bold')
    plt.savefig(save_file)
    plt.close()


def e4_crp_fig(data_to_use, which_list, save_name):
    if which_list == 0:
        ylims_tf = [-.025, .2]
        ylims_crp = [0., .2]
    elif which_list == 1:
        ylims_tf = [-.025, .3]
        ylims_crp = [0., .3]
    else:
        print("WTF?")
        return

    # make a dummy code for the conditions we want
    data_to_use['dummy_cond'] = data_to_use.task_condition + data_to_use.recall_instruction_condition

    # # give them nice names
    # data_to_use.dummy_cond[data_to_use.dummy_cond == "Varying SizeFree"] = "Varying--Free"
    # data_to_use.dummy_cond[data_to_use.dummy_cond == "Constant SizeFree"] = "Constant--Free"
    # data_to_use.dummy_cond[data_to_use.dummy_cond == "Constant SizeSerial"] = "Constant--Serial"

    # get rid of varying serial
    data_to_use = data_to_use[data_to_use.dummy_cond != "Varying SizeSerial"]

    # setup the grid
    fig2 = plt.figure(figsize=(two_col, two_col/2))
    gs = gridspec.GridSpec(1, 2)
    crp_axis = fig2.add_subplot(gs[0, 0])
    tf_axis = fig2.add_subplot(gs[0, 1])

    # plot crps
    data_filter = data_to_use.list == which_list
    rcParams['lines.linewidth'] = 1
    rcParams['lines.markersize'] = 0
    g = sns.factorplot(x="lag", y="crp", hue="dummy_cond", data=data_to_use.loc[data_filter, :],  dodge=.25,
                       units='subject', ax=crp_axis,
                       hue_order=["Varying SizeFree", "Constant SizeFree", "Constant SizeSerial"],
                       legend_out=True, handlelength=.01)
    crp_axis.set(xlabel="Lag", ylabel="Cond. Resp. Prob.", ylim=ylims_crp, xticks=range(0, 11, 2),
                 yticks=np.arange(0, ylims_crp[1]+0.05, 0.05),
                 xticklabels=range(-5, 6, 2))

    crp_axis.legend(title='Condition', ncol=1, labelspacing=.2,  loc=2)
    plt.figure(fig2.number)
    sns.despine()
    crp_axis.annotate('A.', xy=(-.21, 1), xycoords='axes fraction', weight='bold')

    # plot temp factors
    data_filter = np.logical_and(data_to_use.list == which_list, data_to_use.lag == 0)
    g = sns.barplot(x="dummy_cond", y=tf_col, data=data_to_use.loc[data_filter, :],
                    order=["Varying SizeFree", "Constant SizeFree", "Constant SizeSerial"], ax=tf_axis)
    tf_axis.set(xlabel="Condition", ylabel="z(TCE)", ylim=ylims_tf,
                xticklabels=["Varying-Free", "Constant-Free", "Constant-Serial"],
                yticks=np.arange(0, ylims_tf[1]+0.05, 0.05)) # set_xticklabels
    tf_axis.lines[0].set_color('grey')
    tf_axis.lines[1].set_color('black')
    tf_axis.lines[2].set_color('black')
    plt.axhline(linewidth=1, linestyle='--', color='k')
    plt.figure(fig2.number)
    sns.despine()
    tf_axis.annotate('B.', xy=(-.19, 1), xycoords='axes fraction', weight='bold')

    fig2.savefig(save_name + '.pdf', bbox_inches='tight')
    plt.close(fig2)


def corr_fig(all_crps, save_name):
    def extended(ax, x, y, **args):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_ext = np.linspace(xlim[0], xlim[1], 100)
        p = np.polyfit(x, y , deg=1)
        y_ext = np.poly1d(p)(x_ext)
        ax.plot(x_ext, y_ext, **args)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax

    # Load in the data
    all_crps.to_csv('new_data.csv')
    data = pd.read_csv('new_data.csv')

    # Filter out unnecessary data
    data = data[data['lag'] == 0]
    data = data[data['list'] == 0]

    # Number all the conditions to numbers one by one
    data.ix[((data['task_condition'] == 'Shoebox') & (data['instruction_condition'] == 'Explicit') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 0
    data.ix[((data['task_condition'] == 'Shoebox') & (data['instruction_condition'] == 'Incidental') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 1

    data.ix[((data['task_condition'] == 'Front Door') & (data['instruction_condition'] == 'Explicit') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 2
    data.ix[((data['task_condition'] == 'Front Door') & (data['instruction_condition'] == 'Incidental') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 3

    data.ix[((data['task_condition'] == 'Weight') & (data['instruction_condition'] == 'Incidental') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 4
    data.ix[((data['task_condition'] == 'Animacy') & (data['instruction_condition'] == 'Incidental') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 5
    data.ix[((data['task_condition'] == 'Scenario') & (data['instruction_condition'] == 'Incidental') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 6
    data.ix[((data['task_condition'] == 'Movie') & (data['instruction_condition'] == 'Incidental') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 7
    data.ix[((data['task_condition'] == 'Relational') & (data['instruction_condition'] == 'Incidental') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 8
    data.ix[((data['task_condition'] == 'Varying Size') & (data['instruction_condition'] == 'Incidental') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 9

    data.ix[((data['task_condition'] == 'Constant Size') & (data['instruction_condition'] == 'Incidental') & (
                data['recall_instruction_condition'] == 'Free')), 'condition_num'] = 10
    data.ix[((data['task_condition'] == 'Constant Size') & (data['instruction_condition'] == 'Incidental') & (
                data['recall_instruction_condition'] == 'Serial')), 'condition_num'] = 11

    data.ix[((data['task_condition'] == 'Varying Size') & (data['instruction_condition'] == 'Incidental') & (data['recall_instruction_condition'] == 'Serial')), 'condition_num'] = 12
    data['condition_num'] = data['condition_num'].astype(int) # Make sure they are saved as ints

    # drop the varying size serial data
    data = data[data.condition_num != 12]

    # Get the means of each condition
    means = data.groupby('condition_num').mean()

    # Calculate the regression
    x = means.prec
    y = means.all_tf_z
    slope, intercept, r_value, p_value, std_err = stats.linregress(x ,y)
    print 'r(%d) = %.2f, p = %.2f' % (len(x) - 2, r_value, p_value)

    # Make the figure
    fig1 = plt.figure(figsize=(one_col, one_col))
    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    # here is the new part
    for cond in means.index.values:
        plt.plot(x[cond], y[cond], marker="$" + str(cond + 1) + "$", markersize=8, color='#000000')

    # ax1.scatter(x, y, c='k', s=50)
    ax1.grid(True)
    ax1.set(xlabel="Recall Prob.", ylabel="z(TCE)")

    # ax1.set(xlim=[0.35, 0.5], xticks=[0.35, 0.4, 0.45, 0.5])
    # ax1.set(ylim=[0, 0.14])

    # ax1.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), 'k--', lw = 2)

    # Using this function makes sure that the dotted line runs across the entire figure and is not just a segment
    ax1 = extended(ax1, np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='black',
                   linestyle='dashed', markersize=0, linewidth=1)

    ax1.text(.26, .135, '$\mathit{r}(%d) = %.2f, \mathit{p} = %.2f$' % (len(x) - 2, r_value, p_value), fontsize=10)
    # "%s is %d years old." % (name, age)

    # Save the figure
    plt.savefig(save_name)
    plt.close()

def meta_fig(all_crps, save_file):
    # serial vs free
    c1 = all_crps[np.logical_and(
        np.logical_and(
            np.logical_and(all_crps.task_condition == "Constant Size",
                           all_crps.recall_instruction_condition == "Free"),
            all_crps.list == 0),
        all_crps.lag == 0)].all_tf_z.values
    c2 = all_crps[np.logical_and(
        np.logical_and(
            np.logical_and(all_crps.task_condition == "Varying Size",
                           all_crps.recall_instruction_condition == "Free"),
            all_crps.list == 0),
        all_crps.lag == 0)].all_tf_z.values
    c3 = all_crps[np.logical_and(
        np.logical_and(
            np.logical_and(all_crps.task_condition == "Constant Size",
                           all_crps.recall_instruction_condition == "Serial"),
            all_crps.list == 0),
        all_crps.lag == 0)].all_tf_z.values
    print((c1.shape, c2.shape))
    print((c1[~np.isnan(c1)].shape, c2[~np.isnan(c2)].shape))
    print(ttest_ind(c1[~np.isnan(c1)], c2[~np.isnan(c2)]))
    print(ttest_ind(c1[~np.isnan(c1)], c3[~np.isnan(c3)]))
    print(ttest_ind(c2[~np.isnan(c2)], c3[~np.isnan(c3)]))

    # list 1 vs list 2 for explicit
    c1 = all_crps[np.logical_and(
        np.logical_and(
            np.logical_and(all_crps.task_condition == "Shoebox",
                           all_crps.instruction_condition == "Explicit"),
            all_crps.list == 0),
        all_crps.lag == 0)].all_tf_z.values
    c2 = all_crps[np.logical_and(
        np.logical_and(
            np.logical_and(all_crps.task_condition == "Shoebox",
                           all_crps.instruction_condition == "Explicit"),
            all_crps.list == 1),
        all_crps.lag == 0)].all_tf_z.values
    ttest_ind(c1[~np.isnan(c1)], c2[~np.isnan(c2)])

    # list 1 vs list 2 for explicit
    c1 = all_crps[np.logical_and(
        np.logical_and(
            np.logical_and(all_crps.task_condition == "Shoebox",
                           all_crps.instruction_condition == "Incidental"),
            all_crps.list == 0),
        all_crps.lag == 0)].all_tf_z.values
    c2 = all_crps[np.logical_and(
        np.logical_and(
            np.logical_and(all_crps.task_condition == "Shoebox",
                           all_crps.instruction_condition == "Incidental"),
            all_crps.list == 1),
        all_crps.lag == 0)].all_tf_z.values
    ttest_ind(c1[~np.isnan(c1)], c2[~np.isnan(c2)])

    these_conds = ["Movie", "Scenario", "Animacy", "Weight"]
    ds = []
    for cond in these_conds:
        c = all_crps[np.logical_and(
            np.logical_and(
                np.logical_and(all_crps.task_condition == cond,
                               all_crps.instruction_condition == "Incidental"),
                all_crps.list == 0),
            all_crps.lag == 0)].all_tf_z
        ds.append(c.mean() / c.std())
    mean_ds = np.mean(ds)
    std_ds = np.std(ds, ddof=1)
    CI_ds = std_ds / np.sqrt(len(ds)) * 1.96
    pm = [mean_ds - CI_ds, mean_ds + CI_ds]

    these_conds = ["Shoebox", "Front Door"]
    ds = []
    for cond in these_conds:
        c = all_crps[np.logical_and(
            np.logical_and(
                np.logical_and(all_crps.task_condition == cond,
                               all_crps.instruction_condition == "Incidental"),
                all_crps.list == 0),
            all_crps.lag == 0)].all_tf_z
        ds.append(c.mean() / c.std())
    mean_ds = np.mean(ds)
    std_ds = np.std(ds)

    these_conds = ["Movie", "Scenario", "Animacy", "Weight", "Shoebox", "Front Door", "Constant Size", "Varying Size"]
    ds = []
    for cond in these_conds:
        c = all_crps[np.logical_and(np.logical_and(
            np.logical_and(
                np.logical_and(all_crps.task_condition == cond,
                               all_crps.instruction_condition == "Incidental"),
                all_crps.list == 0),
            all_crps.lag == 0), all_crps.recall_instruction_condition == "Free")].all_tf_z
        ds.append(c.mean() / c.std())
    mean_ds = np.mean(ds)
    std_ds = np.std(ds, ddof=1)
    CI_ds = std_ds / np.sqrt(len(ds)) * 1.96
    pm = [mean_ds - CI_ds, mean_ds + CI_ds]

    these_conds = ["Shoebox", "Front Door", "Weight", "Animacy", "Scenario", "Movie", "Constant Size", "Varying Size"]
    these_conds = list(reversed(these_conds))
    ds_m = np.zeros((1, len(these_conds) + 1)) * np.nan
    ds_e = np.zeros((2, len(these_conds) + 1)) * np.nan
    for cond_n, cond in enumerate(these_conds):
        c = all_crps[np.logical_and(np.logical_and(
            np.logical_and(
                np.logical_and(all_crps.task_condition == cond,
                               all_crps.instruction_condition == "Incidental"),
                all_crps.list == 0),
            all_crps.lag == 0), all_crps.recall_instruction_condition == "Free")].all_tf_z
        ss = np.arange(c.shape[0])
        permutation_distribution = np.zeros((10000, 1)) * np.nan
        for perm_i in np.arange(10000):
            these = c.values[np.random.choice(ss, ss.shape[0])]
            permutation_distribution[perm_i, :] = np.nanmean(these) / np.nanstd(these)
        bot = np.percentile(permutation_distribution, 2.5, axis=0)
        top = np.percentile(permutation_distribution, 97.5, axis=0)
        m = c.mean() / c.std()
        e = np.stack((top - m, m - bot))
        ds_m[:, cond_n + 1] = m
        ds_e[[0, 1], cond_n + 1] = np.squeeze(e)

    mean_ds = np.nanmean(ds_m)
    std_ds = np.nanstd(ds_m, ddof=1)
    CI_ds = std_ds / np.sqrt(len(these_conds)) * 2.365  # (t_critical with df = 7)
    ds_e[:, 0] = [CI_ds, CI_ds]
    ds_m[:, 0] = mean_ds

    one_col = 3.5
    two_col = one_col * 2
    base_height = 2.5
    plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
    fig = plt.figure(figsize=(one_col, base_height * 2))
    lines = {'linestyle': 'None'}
    plt.rc('lines', **lines)
    plt.errorbar(y=np.arange(len(these_conds)) + 1, x=ds_m[:, 1:].T, xerr=ds_e[:, 1:], marker='o')
    plt.errorbar(y=0, x=ds_m[:, 0], xerr=np.expand_dims(ds_e[:, 0], axis=1), marker='D', markersize=7, color='k')
    these_conds.insert(0, 'Average')
    plt.gca().set(yticks=np.arange(len(these_conds)), yticklabels=these_conds, ylim=[-1, len(these_conds)])
    plt.axvline(linewidth=1, linestyle='--', color='k', markersize=0)
    plt.xlabel("Cohen's $d$")
    plt.ylabel('Condition')
    plt.savefig(save_file)


def model_it(list0):
    parameters = [.1, .1, .1, .1, .1, .1, .1, .1]

    conditions = (
        ('Explicit', 'Shoebox', 'Free'),
        ('Incidental', 'Shoebox', 'Free'),
        ('Explicit', 'Front Door', 'Free'),
        ('Incidental', 'Front Door', 'Free'),
        ('Incidental', 'Movie', 'Free'),
        ('Incidental', 'Relational', 'Free'),
        ('Incidental', 'Scenario', 'Free'),
        ('Incidental', 'Animacy', 'Free'),
        ('Incidental', 'Weight', 'Free'),
        ('Incidental', 'Constant Size', 'Free'),
        ('Incidental', 'Constant Size', 'Serial'),
        ('Incidental', 'Varying Size', 'Free'),
        ('Incidental', 'Varying Size', 'Serial'),
    )

    runs_per_param_set = 1000
    n_lists = 1
    n_items = 16
    pop_size = 300
    gens_per_ss = 30
    polish = True
    n_final_runs = 1000
    bounds = [
        (1.0, 5.0),  # 0: phi_s
        (0.1, 3.0),  # 1: phi_d
        (0.0, .99),  # 2: gamma_fc
        (0.0, .99),  # 2: gamma_cf
        (0.0, .3),  # 3: beta_enc
        (0.0, 0.8),  # 4: theta_s
        (0.0, 0.8),  # 5: theta_r
        (1.0, 3.0),  # 6: tau
        (0.00, .5),  # 7: beta_rec
        (0.00, .99)  # 7: beta_drift
    ]

    output = []
    for cond in conditions:
        filter = np.logical_and(list0.instruction_condition == cond[0],
                                np.logical_and(list0.task_condition == cond[1],
                                               np.logical_and(list0.recall_instruction_condition == cond[2],
                                                              list0.list == 0)))
        data_vector = np.append(np.nanmean(cbcc.prec(list0.recalls[filter].values, n_items)),
                                np.nanmean(cbcc.temporal_factor(list0.recalls[filter].values, n_items)))

        args = (runs_per_param_set, n_lists, n_items, data_vector)
        result = differential_evolution(tcm.evaluate, bounds, args,
                                        polish=polish, maxiter=gens_per_ss, popsize=pop_size, disp=False)
        recalled_items = tcm.tcm(result.x, n_final_runs, n_lists, n_items)
        model_vector = np.append(np.nanmean(cbcc.prec(recalled_items.astype('int64'), n_items)),
                                 np.nanmean(cbcc.temporal_factor(recalled_items.astype('int64'), n_items)))
        output.append((cond, result, data_vector, model_vector))
        print(cond, result.x)

    with open('parrot.pkl', 'wb') as f:
        pickle.dump(output, f)

    with open('parrot.pkl', 'rb') as f:
        loaded = pickle.load(f)

    out = 1
    plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
    fig = plt.figure(figsize=(7, 7))
    for cond in loaded:
        plt.plot(cond[-2][0], cond[-2][1], marker="$%d$" % out, markersize=20, color='#000000')
        plt.plot(cond[-1][0], cond[-1][1], marker="$%d$" % out, markersize=20, color='#808080')
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)

        out += 1
    plt.ylabel('Temporal Factor Score')
    plt.xlabel('Probability of Recall')
    plt.savefig('fits.pdf')