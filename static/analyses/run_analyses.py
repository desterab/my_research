import sys
sys.path.append("/home/khealey/code/py_modules/")
from beh_tools import recall_dynamics as rdf
import os
import pickle
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns



recalls = pickle.load(open("/Users/khealey/code/experiments/Heal16implicit/HealEtal16implicit.data.pkl", "rb"))

rec_mat = recalls.as_matrix(range(recalls.shape[1]-2))

# # compute a lag-CRP
# crp = rdf. crp(listlen=16, recalls=rec_mat, filter_ind=None, allow_repeats=False, exclude_op=0)
#
# # make it a dataframe
# crp = pd.DataFrame(crp)

# loop over subjects and lists, for each isolate their data
subjects = recalls.subject.unique()
lists = recalls.list.unique()
all_crps = pd.DataFrame()
for s in subjects:
    for l in lists:
        s_filter = recalls.subject == s
        l_filter = recalls.list == l
        cur_recalls = recalls.loc[s_filter & l_filter, :]
        if cur_recalls.shape[0] == 0:
            continue
        rec_mat = cur_recalls.as_matrix(range(recalls.shape[1] - 2))
        crp = rdf.crp(listlen=16, recalls=rec_mat, filter_ind=None, allow_repeats=False, exclude_op=0)
        crp = pd.DataFrame.from_records(crp)

        # skip lists with zero recall accuracy
        prec = rdf.prec(listlen=16, recalls=rec_mat)
        if prec <= 0:
            continue

        # skip lists if subject reported being aware
        if cur_recalls.aware[0] == 'yes':
            continue

        # add list number, and condition ids
        crp['subject'] = pd.Series([s for x in range(len(crp.index))], index=crp.index)
        crp['list'] = pd.Series([l for x in range(len(crp.index))], index=crp.index)
        crp['instruction_condition'] = pd.Series([cur_recalls.instruction_condition[0] for x in range(len(crp.index))], index=crp.index)
        crp['task_condition'] = pd.Series([cur_recalls.task_condition[0] for x in range(len(crp.index))], index=crp.index)
        crp['prec'] =  pd.Series([prec for x in range(len(crp.index))], index=crp.index)

        all_crps = pd.concat([all_crps, crp])


g = sns.factorplot(x="lag", y="crp", hue="task_condition", col="list", row="instruction_condition", data=all_crps.loc[all_crps.lag.abs() <= 5, :])
print pd.crosstab(all_crps.lag,[all_crps.instruction_condition, all_crps.task_condition])

# 405  393  402  380  387  380

