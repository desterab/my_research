import anal_funcs as af
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rcParams
from cbcc_tools.beh_anal import recall_dynamics as cbcc

# params for data prep and saving results
results_dir = "../dissemination/manuscript/jml/second_submission/figures/"
dict_path = "/Users/khealey/code/py_modules/cbcc_tools/wordpool_files/websters_dict.txt"

# number of permutations for permutations tests
n_perms = 1

# load or make the recall matrix
recalls = af.make_psiturk_recall_matrix(remake_data_file=False, dict_path=dict_path,
                                        save_file='HealEtal16implicit.recalls')

# load or compute the recall dynamics
all_crps = af.load_the_data(n_perms=n_perms, remake_data_file=False,
                            recalls_file='HealEtal16implicit.recalls.pkl', save_name=results_dir)

# convert to xarray to make compatible with cbcc_tools --- then run RDF analyses
all_data = pd.DataFrame.from_csv('/Users/khealey/code/experiments/Heal16implicit/dissemination/manuscript/jml/second_submission/figures/Heal16implicit_data.csv')
list0, sample_sizes_aware_counts, sample_sizes_included_counts = af.make_xarray(all_data.copy(deep=True), 0)
list1, sample_sizes_aware_counts, sample_sizes_included_counts = af.make_xarray(all_data.copy(deep=True), 1)
cbcc.run_these_analyses(list0, ['pfr', 'spc', 'lag_crp'])
cbcc.run_these_analyses(list1, ['pfr', 'spc', 'lag_crp'])


# make figures etc
plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')

# make table 1
all_crps = af.sample_size_table(all_crps, results_dir)

# Make a few plots of the CRPs generate by the cbcc_tools code just as a check to ensure it is doing exactly
# the same thing as the legacy code used for the main CRP figures in the paper
e1_explicit_filter = np.logical_and(list0.instruction_condition == 'Explicit', list0.task_condition == 'Shoebox')
e1_implicit_filter = np.logical_and(list0.instruction_condition == 'Incidental', list0.task_condition == 'Shoebox')
cbcc.lag_crp_plot(list0.lag_crp[e1_explicit_filter])
cbcc.lag_crp_plot(list0.lag_crp[e1_implicit_filter])
plt.ylim(0, .2)
plt.savefig('e1.pdf')
plt.close()
e2_explicit_filter = np.logical_and(list0.instruction_condition == 'Explicit', list0.task_condition == 'Front Door')
e2_implicit_filter = np.logical_and(list0.instruction_condition == 'Incidental', list0.task_condition == 'Front Door')
cbcc.lag_crp_plot(list0.lag_crp[e2_explicit_filter])
cbcc.lag_crp_plot(list0.lag_crp[e2_implicit_filter])
plt.ylim(0, .2)
plt.savefig('e2.pdf')
plt.close()



####### make E3 Figures

# spc/pfr for list 0
plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')



af.e3_spc_fig(list0, results_dir + "E3_spc_list1")
af.e3_spc_fig(list1, results_dir + "E3_spc_list2")



# # spc/pfr for list 0
# fig = plt.figure(figsize=(9, 8))
# gs = gridspec.GridSpec(2, 5)
# instruction_cond_filter = to_plot.instruction_condition == "Incidental"
# task_list = ["Weight", "Animacy",  "Scenario", "Movie", "Relational"]
# task_col = 0
# for task in task_list:
#     spc_axis = fig.add_subplot(gs[0, task_col])
#     cbcc.spc_plot(to_plot.spc[np.logical_and(instruction_cond_filter, to_plot.task_condition == task)], ax=spc_axis)
#     spc_axis.set_title(task)
#     spc_axis.set_xlabel('')
#     spc_axis.get_xaxis().set_ticklabels([])
#     pfr_axis = fig.add_subplot(gs[1, task_col])
#     cbcc.pfr_plot(to_plot.pfr[np.logical_and(instruction_cond_filter, to_plot.task_condition == task)], ax=pfr_axis)
#     pfr_axis.set_ylim(0, 0.5)
#     if task_col > 0:
#         spc_axis.set_ylabel('')
#         spc_axis.get_yaxis().set_ticklabels([])
#         pfr_axis.set_ylabel('')
#         pfr_axis.get_yaxis().set_ticklabels([])
#     task_col += 1
# plt.savefig(results_dir + "E3_spc_list1")
# plt.close()


















####### make E1 Figures

# spc/pfr for list 0
plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
af.spc_encoding_instructions_fig(list0, 'Shoebox', results_dir + 'E1_spc_list1.pdf')
af.spc_encoding_instructions_fig(list1, 'Shoebox', results_dir + 'E1_spc_list2.pdf')

rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 0
which_list = 0
data_filter = np.logical_and(all_crps.task_condition == "Shoebox", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E1_crp_list1")

which_list = 1
data_filter = np.logical_and(all_crps.task_condition == "Shoebox", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E1_crp_list2")


####### make E2 Figures

# spc/pfr for list 0
plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
af.spc_encoding_instructions_fig(list0, 'Front Door', results_dir + 'E2_spc_list1.pdf')
af.spc_encoding_instructions_fig(list1, 'Front Door', results_dir + 'E2_spc_list2.pdf')

rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 0
which_list = 0
data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E2_crp_list1")

which_list = 1
data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E2_crp_list2")









#
#
#
#
#
# # make E2 Figures
# which_list = 0
# data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E2_crp_list1")
#
# which_list = 1
# data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E2_crp_list2")
#
#
#
#
#
#
# # make E3 Figures
# which_list = 0
# instruction_cond_filter = all_crps.instruction_condition == "Incidental"
# task_cond_filter = all_crps.task_condition.isin(["Movie", "Relational", "Scenario", "Animacy", "Weight"])
# recall_cond_filter = all_crps.recall_instruction_condition == "Free"
# lag_filter = all_crps.lag.abs() <= 5
# list_filter = all_crps.list == which_list
# data_filter = np.logical_and(instruction_cond_filter,
#                              np.logical_and(task_cond_filter,
#                                             np.logical_and(lag_filter, list_filter)))
# data_to_use = all_crps.loc[data_filter, :]
# af.e3fig(data_to_use, results_dir + "E3_crp_list1")
#
# which_list = 1
# instruction_cond_filter = all_crps.instruction_condition == "Incidental"
# task_cond_filter = all_crps.task_condition.isin(["Movie", "Relational", "Scenario", "Animacy", "Weight"])
# recall_cond_filter = all_crps.recall_instruction_condition == "Free"
# lag_filter = all_crps.lag.abs() <= 5
# list_filter = all_crps.list == which_list
# data_filter = np.logical_and(instruction_cond_filter,
#                              np.logical_and(task_cond_filter,
#                                             np.logical_and(lag_filter, list_filter)))
# data_to_use = all_crps.loc[data_filter, :]
# af.e3fig(data_to_use, results_dir + "E3_crp_list2")
#
#
#
#
#
#
# # af.processing_task_fig(data_to_use, which_instruction_cond, which_list, results_dir + 'E3')
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
# #
# #
# #
# # # make crp/temp fact figure for constant exp
# # which_list = 0
# # data_filter = np.logical_and(all_crps.task_condition == "Constant Size", all_crps.lag.abs() <= 5)
# # data_to_use = all_crps.loc[data_filter, :]
# # af.E4_fig(data_to_use, which_list, results_dir + "E4_constant")
# #
# # # make crp/temp fact figure for varying exp
# # which_list = 0
# # data_filter = np.logical_and(all_crps.task_condition == "Varying Size", all_crps.lag.abs() <= 5)
# # data_to_use = all_crps.loc[data_filter, :]
# # af.E4_fig(data_to_use, which_list, results_dir + "E4_varying")
