import os
import pickle
from anal_funcs import *


remake_data_file = True


# load or create the data
if os.path.isfile("all_crps.pkl") and not remake_data_file:
    all_crps = pickle.load(open("all_crps.pkl", "rb"))
    all_spcs = pickle.load(open("all_spcs.pkl", "rb"))
else:
    all_crps, all_spcs = load_the_data(n_perms=1)
    all_crps.to_pickle("all_crps.pkl")
    all_spcs.to_pickle("all_spcs.pkl")

# take mean within subject and lag, check sample sizes
all_crps = all_crps.groupby(['subject', 'lag', 'list'], as_index=False).mean()
all_spcs = all_spcs.groupby(['subject', 'serial_pos', 'list'], as_index=False).mean()
all_crps["from_chance"] = all_crps["all_tf"] - .5

print pd.crosstab(all_crps[np.logical_and(all_crps.lag==0,all_crps.list==0)].lag, [all_crps.instruction_condition, all_crps.task_condition])


all_crps.loc[all_crps.task_condition == 0, 'task_condition'] = "Size"
all_crps.loc[all_crps.task_condition == 1, 'task_condition'] = "Deep"
all_crps.loc[all_crps.task_condition == 2, 'task_condition'] = "Relational"
all_crps.loc[all_crps.task_condition == 3, 'task_condition'] = "Scenario"

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


##### Figure 1
which_list = 0
data_filter = np.logical_and(all_crps.task_condition == "Scenario", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
encoding_instruct_fig(data_to_use, which_list, "Figure1")

##### Figure 2
which_instruction_cond = "Implicit"
which_list = 0
data_filter = np.logical_and(all_crps.task_condition != "Relational", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
processing_task_fig(data_to_use, which_instruction_cond, which_list, 'figure2')

##### Figure 3
which_instruction_cond = "Explicit"
which_list = 0
data_filter = np.logical_and(all_crps.task_condition != "Relational", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
processing_task_fig(data_to_use, which_instruction_cond, which_list, 'figure3')


######Figure 4
which_list = 0
data_filter = np.logical_and(all_crps.task_condition != "Relational", all_crps.lag.abs() == 0)
data_to_use = all_crps.loc[data_filter, :]
prec_fig(data_to_use, which_list, "figure4")








#################### figures for list 2

##### Figure 1
which_list = 1
data_filter = np.logical_and(all_crps.task_condition == "Scenario", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
encoding_instruct_fig(data_to_use, which_list, "l2_figure1")

##### Figure 2
which_instruction_cond = "Implicit"
which_list = 1
data_filter = np.logical_and(all_crps.task_condition != "Relational", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
processing_task_fig(data_to_use, which_instruction_cond, which_list, 'l2_figure2')

##### Figure 3
which_instruction_cond = "Explicit"
which_list = 1
data_filter = np.logical_and(all_crps.task_condition != "Relational", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
processing_task_fig(data_to_use, which_instruction_cond, which_list, 'l2_figure3')


######Figure 4
which_list = 1
data_filter = np.logical_and(all_crps.task_condition != "Relational", all_crps.lag.abs() == 0)
data_to_use = all_crps.loc[data_filter, :]
prec_fig(data_to_use, which_list, "l2_figure4")




#
# # setup the grid
# fig2 = plt.figure(figsize=(30, 10))
# gs = gridspec.GridSpec(1, 2)
# crp_axis = fig2.add_subplot(gs[0, 0])
# tf_axis = fig2.add_subplot(gs[0, 1])
#
# # plot crps
# data_filter = np.logical_and(data_to_use.instruction_condition == which_cond, data_to_use.list == which_list)
# sns.factorplot(x="lag", y="crp", hue="task_condition", data=data_to_use.loc[data_filter, :], ax=crp_axis, hue_order=["Scenario", "Size", "Deep"])
# crp_axis.set(xlabel="Lag", ylabel="Cond. Resp. Prob.", ylim=[0., .2])
# crp_axis.legend(title='Processing Task')
# plt.figure(fig2.number)
# sns.despine()
#
# # plot temp factors
# data_filter = np.logical_and(data_to_use.instruction_condition == which_cond, data_to_use.list == which_list)
# sns.barplot(x="task_condition", y='all_tf', data=data_to_use.loc[data_filter, :], ax=tf_axis, order=["Scenario", "Size", "Deep"])
# tf_axis.set(xlabel="Processing Task", ylabel="Temporal Factor", ylim=[.5, .6])
# plt.figure(fig2.number)
# sns.despine()
#
# fig2.savefig('figure2' + '.pdf', bbox_inches='tight')
# plt.close(fig2)














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
#
