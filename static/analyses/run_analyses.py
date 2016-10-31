import os
import pickle
from anal_funcs import *


remake_data_file = False


# load or create the data
if os.path.isfile("all_crps.pkl") and not remake_data_file:
    all_crps = pickle.load(open("all_crps.pkl", "rb"))
    all_spcs = pickle.load(open("all_spcs.pkl", "rb"))
else:
    all_crps, all_spcs = load_the_data(n_perms=80)
    all_crps.to_pickle("all_crps.pkl")
    all_spcs.to_pickle("all_spcs.pkl")

# take mean within subject and lag, check sample sizes
all_crps = all_crps.groupby(['subject', 'lag', 'list'], as_index=False).mean()
all_spcs = all_spcs.groupby(['subject', 'serial_pos', 'list'], as_index=False).mean()
all_crps["from_chance"] = all_crps["all_tf"] - .5

print pd.crosstab(all_crps[np.logical_and(all_crps.lag==0,all_crps.list==0)].lag, [all_crps.instruction_condition, all_crps.task_condition])


all_crps.loc[all_crps.task_condition == 0, 'task_condition'] = "Shoebox"
all_crps.loc[all_crps.task_condition == 1, 'task_condition'] = "Movie"
all_crps.loc[all_crps.task_condition == 2, 'task_condition'] = "Relational"
all_crps.loc[all_crps.task_condition == 3, 'task_condition'] = "Scenario"
all_crps.loc[all_crps.task_condition == 4, 'task_condition'] = "Animacy"
all_crps.loc[all_crps.task_condition == 5, 'task_condition'] = "Weight"
all_crps.loc[all_crps.task_condition == 6, 'task_condition'] = "Front Door"


all_crps.loc[all_crps.instruction_condition == 0, 'instruction_condition'] = "Explicit"
all_crps.loc[all_crps.instruction_condition == 1, 'instruction_condition'] = "Implicit"


# overall figure style
# sns.set_style("white")
# sns.set_style("whitegrid")
sns.set_style("ticks")
sns.set_context("talk", font_scale=2.5, rc={"lines.linewidth": 4})
colors = ["#000000", "#808080", "#D3D3D3"]  # black and white
sns.set_palette(colors)
# sns.set_palette(sns.dark_palette("grey", n_colors=3))



# isolate the conditions we want to plot


# ##### shoebox
# which_list = 0
# data_filter = np.logical_and(all_crps.task_condition == "Shoebox", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# encoding_instruct_fig(data_to_use, which_list, "shoebox")
#
# ##### door
# which_list = 0
# data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# encoding_instruct_fig(data_to_use, which_list, "door")


##### many tasks
which_instruction_cond = "Implicit"
which_list = 0
data_filter = np.logical_and(np.logical_and(all_crps.task_condition != "Shoebox", all_crps.task_condition != "Front Door"), all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
processing_task_fig(data_to_use, which_instruction_cond, which_list, 'E3')

##### Figure 3
which_instruction_cond = "Explicit"
which_list = 0
data_filter = np.logical_and(all_crps.task_condition != "Size", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
processing_task_fig(data_to_use, which_instruction_cond, which_list, 'figure3')


######Figure 4
which_list = 0
data_filter = all_crps.lag.abs() == 0
data_to_use = all_crps.loc[data_filter, :]
prec_fig(data_to_use, which_list, "figure4")





#
#
#
# #################### figures for list 2
#
# ##### Figure 1
# which_list = 1
# data_filter = np.logical_and(all_crps.task_condition == "Scenario", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# encoding_instruct_fig(data_to_use, which_list, "l2_figure1")
#
# ##### Figure 2
# which_instruction_cond = "Implicit"
# which_list = 1
# data_filter = np.logical_and(all_crps.task_condition != "Relational", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# processing_task_fig(data_to_use, which_instruction_cond, which_list, 'l2_figure2')
#
# ##### Figure 3
# which_instruction_cond = "Explicit"
# which_list = 1
# data_filter = np.logical_and(all_crps.task_condition != "Relational", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# processing_task_fig(data_to_use, which_instruction_cond, which_list, 'l2_figure3')
#
#
# ######Figure 4
# which_list = 1
# data_filter = np.logical_and(all_crps.task_condition != "Relational", all_crps.lag.abs() == 0)
# data_to_use = all_crps.loc[data_filter, :]
# prec_fig(data_to_use, which_list, "l2_figure4")
#
#
#
#
#
#
#
#
#
#
#
#
#

#
#
#
# # make the exploratory figures
# apply_perm_correction = False
# if apply_perm_correction:
#     fig_prefix = "_perm"
# else:
#     fig_prefix = "_standard"
# fig_compare_tasks(all_crps, all_spcs, which_cond=0, which_list=0, apply_perm_correction=apply_perm_correction, print_to="explicit" + fig_prefix + "_l0")
# fig_compare_tasks(all_crps, all_spcs, which_cond=0, which_list=1, apply_perm_correction=apply_perm_correction, print_to='explicit' + fig_prefix + '_l1')
# fig_compare_tasks(all_crps, all_spcs, which_cond=1, which_list=0, apply_perm_correction=apply_perm_correction, print_to='implicit' + fig_prefix + '_l0')
# fig_compare_tasks(all_crps, all_spcs, which_cond=1, which_list=1, apply_perm_correction=apply_perm_correction, print_to='implicit' + fig_prefix + '_l1')

