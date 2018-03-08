import anal_funcs as af
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rcParams
from cbcc_tools.beh_anal import recall_dynamics as cbcc
import seaborn as sns
import tcm_Heal16implicit as tcm
from scipy.optimize import differential_evolution
import pickle

# params for data prep and saving results
results_dir = "../dissemination/manuscript/jml/second_submission/figures/"
dict_path = "/Users/khealey/code/py_modules/cbcc_tools/wordpool_files/websters_dict.txt"

# number of permutations for permutations tests
n_perms = 10000

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










#
#
# # Make a few plots of the CRPs generate by the cbcc_tools code just as a check to ensure it is doing exactly
# # the same thing as the legacy code used for the main CRP figures in the paper
# e1_explicit_filter = np.logical_and(list0.instruction_condition == 'Explicit', list0.task_condition == 'Shoebox')
# e1_implicit_filter = np.logical_and(list0.instruction_condition == 'Incidental', list0.task_condition == 'Shoebox')
# cbcc.lag_crp_plot(list0.lag_crp[e1_explicit_filter])
# cbcc.lag_crp_plot(list0.lag_crp[e1_implicit_filter])
# plt.ylim(0, .2)
# plt.savefig('e1.pdf')
# plt.close()
# e2_explicit_filter = np.logical_and(list0.instruction_condition == 'Explicit', list0.task_condition == 'Front Door')
# e2_implicit_filter = np.logical_and(list0.instruction_condition == 'Incidental', list0.task_condition == 'Front Door')
# cbcc.lag_crp_plot(list0.lag_crp[e2_explicit_filter])
# cbcc.lag_crp_plot(list0.lag_crp[e2_implicit_filter])
# plt.ylim(0, .2)
# plt.savefig('e2.pdf')
# plt.close()
#
# ####### general discussion figures
#
# af.corr_fig(all_crps, results_dir + "correlation.pdf")
#
# ##### E4 figs
#
# # spc/pfr
# plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
# af.e4_spc_fig(list0, results_dir + "E4_spc_list1")
#
# # crps
# plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
# af.e4_crp_fig(all_crps, results_dir + "E4_crp_list1")
#
#
# ####### make E1 Figures
#
# # spc/pfr
# plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
# af.spc_encoding_instructions_fig(list0, 'Shoebox', results_dir + 'E1_spc_list1.pdf')
# af.spc_encoding_instructions_fig(list1, 'Shoebox', results_dir + 'E1_spc_list2.pdf')
#
# # list 0 crp
# rcParams['lines.linewidth'] = 1.5
# rcParams['lines.markersize'] = 0
# which_list = 0
# data_filter = np.logical_and(all_crps.task_condition == "Shoebox", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E1_crp_list1")
#
# # list 1 crp
# which_list = 1
# data_filter = np.logical_and(all_crps.task_condition == "Shoebox", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E1_crp_list2")
#
# ####### make E2 Figures
#
# # spc/pfr for list 0
# plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
# af.spc_encoding_instructions_fig(list0, 'Front Door', results_dir + 'E2_spc_list1.pdf')
# af.spc_encoding_instructions_fig(list1, 'Front Door', results_dir + 'E2_spc_list2.pdf')
#
# # list 0 crp
# rcParams['lines.linewidth'] = 1.5
# rcParams['lines.markersize'] = 0
# which_list = 0
# data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E2_crp_list1")
#
# # list 1 crp
# which_list = 1
# data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E2_crp_list2")
#
# ####### make E3 Figures
# plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
#
# # spc/pfr
# rcParams['lines.linewidth'] = 2
# rcParams['lines.markersize'] = 4
# af.e3_spc_fig(list0, results_dir + "E3_spc_list1")
# af.e3_spc_fig(list1, results_dir + "E3_spc_list2")
#
# # list 0 crp
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
# # list 1 crp
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
