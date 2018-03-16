import anal_funcs as af
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cbcc_tools.beh_anal import recall_dynamics as cbcc

# params for data prep and saving results
results_dir = "../dissemination/manuscript/jml/second_submission/figures/"
dict_path = "/Users/khealey/code/py_modules/cbcc_tools/wordpool_files/websters_dict.txt"

# number of permutations for permutations tests
n_perms = 10000

# load or make the recall matrix
recalls = af.make_psiturk_recall_matrix(remake_data_file=False, dict_path=dict_path,
                                        save_file='HealEtal16implicit.recalls')

# load or compute the recall dynamics
all_crps = af.load_the_data(n_perms=n_perms, remake_data_file=True,
                            recalls_file='HealEtal16implicit.recalls.pkl', save_name=results_dir)

# convert to xarray to make compatible with cbcc_tools --- then run RDF analyses
all_data = pd.DataFrame.from_csv('/Users/khealey/code/experiments/Heal16implicit/dissemination/manuscript/jml/'
                                 'second_submission/figures/Heal16implicit_data.csv')
list0, sample_sizes_aware_counts, sample_sizes_included_counts = af.make_xarray(all_data.copy(deep=True), 0)
list1, sample_sizes_aware_counts, sample_sizes_included_counts = af.make_xarray(all_data.copy(deep=True), 1)
cbcc.run_these_analyses(list0, ['pfr', 'spc', 'lag_crp'])
cbcc.run_these_analyses(list1, ['pfr', 'spc', 'lag_crp'])

# make figures etc
which_figures = [1, 2, 3, 4]

# make table 1
all_crps = af.sample_size_table(all_crps, results_dir, recalls)

####### general discussion figures
plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
af.corr_fig(all_crps, results_dir + "correlation.pdf")
af.meta_fig(all_crps, results_dir + "meta.pdf")

####### make E1 Figures
if 1 in which_figures:
    # spc/pfr
    plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
    af.spc_encoding_instructions_fig(list0, 'Shoebox', results_dir + 'E1_spc_list1.pdf')
    af.spc_encoding_instructions_fig(list1, 'Shoebox', results_dir + 'E1_spc_list2.pdf')

    # list 0 crp
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 0
    which_list = 0
    data_filter = np.logical_and(all_crps.task_condition == "Shoebox", all_crps.lag.abs() <= 5)
    data_to_use = all_crps.loc[data_filter, :]
    af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E1_crp_list1")

    # list 1 crp
    which_list = 1
    data_filter = np.logical_and(all_crps.task_condition == "Shoebox", all_crps.lag.abs() <= 5)
    data_to_use = all_crps.loc[data_filter, :]
    af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E1_crp_list2")

####### make E2 Figures
if 2 in which_figures:
    # spc/pfr for list 0
    plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
    af.spc_encoding_instructions_fig(list0, 'Front Door', results_dir + 'E2_spc_list1.pdf')
    af.spc_encoding_instructions_fig(list1, 'Front Door', results_dir + 'E2_spc_list2.pdf')

    # list 0 crp
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 0
    which_list = 0
    data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
    data_to_use = all_crps.loc[data_filter, :]
    af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E2_crp_list1")

    # list 1 crp
    which_list = 1
    data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
    data_to_use = all_crps.loc[data_filter, :]
    af.encoding_instruct_fig(data_to_use, which_list, results_dir + "E2_crp_list2")

##### E4 figs
if 4 in which_figures:
    # spc/pfr
    plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')
    af.e4_spc_fig(list0, results_dir + "E4_spc_list1")

    # list 0 crp
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 0
    which_list = 0
    data_filter = np.logical_and(np.logical_or(all_crps.task_condition == "Constant Size", all_crps.task_condition == "Varying Size"), all_crps.lag.abs() <= 5)
    data_to_use = all_crps.loc[data_filter, :]
    # make a dummy code for the conditions we want
    # x = data_to_use.task_condition + data_to_use.recall_instruction_condition
    # x = x.values
    # data_to_use.loc[:, 'dummy_cond'] = data_to_use.task_condition + data_to_use.recall_instruction_condition

    # give them nice names
    # data_to_use.dummy_cond[data_to_use.dummy_cond == "Varying SizeFree"] = "Varying--Free"
    # data_to_use.dummy_cond[data_to_use.dummy_cond == "Constant SizeFree"] = "Constant--Free"
    # data_to_use.dummy_cond[data_to_use.dummy_cond == "Constant SizeSerial"] = "Constant--Serial"


    # # get rid of varying serial
    # data_to_use = data_to_use[data_to_use.dummy_cond != "Varying SizeSerial"]
    # data_to_use['dummy_cond'].replace("Varying SizeFree", "Varying--Free", inplace=True)
    # data_to_use['dummy_cond'].replace("Constant SizeFree", "Constant--Free", inplace=True)
    # data_to_use['dummy_cond'].replace("Constant SizeSerial", "Constant--Serial", inplace=True)

    this = data_to_use.copy()
    af.e4_crp_fig(this, which_list, results_dir + "E4_crp_list1")

####### make E3 Figures
if 3 in which_figures:
    plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc_bw.mplstyle')

    # spc/pfr
    rcParams['lines.linewidth'] = 2
    rcParams['lines.markersize'] = 4
    af.e3_spc_fig(list0, results_dir + "E3_spc_list1")
    af.e3_spc_fig(list1, results_dir + "E3_spc_list2")

    # list 0 crp
    which_list = 0
    instruction_cond_filter = all_crps.instruction_condition == "Incidental"
    task_cond_filter = all_crps.task_condition.isin(
        ["Movie", "Relational", "Scenario", "Animacy", "Weight", "Varying Size"])
    recall_cond_filter = all_crps.recall_instruction_condition == "Free"
    lag_filter = all_crps.lag.abs() <= 5
    list_filter = all_crps.list == which_list
    data_filter = np.logical_and(instruction_cond_filter,
                                 np.logical_and(task_cond_filter,
                                                np.logical_and(lag_filter, list_filter)))
    data_to_use = all_crps.loc[data_filter, :]
    af.e3fig(data_to_use, results_dir + "E3_crp_list1", which_list)

    # list 1 crp
    which_list = 1
    instruction_cond_filter = all_crps.instruction_condition == "Incidental"
    task_cond_filter = all_crps.task_condition.isin(["Movie", "Relational", "Scenario", "Animacy", "Weight"])
    recall_cond_filter = all_crps.recall_instruction_condition == "Free"
    lag_filter = all_crps.lag.abs() <= 5
    list_filter = all_crps.list == which_list
    data_filter = np.logical_and(instruction_cond_filter,
                                 np.logical_and(task_cond_filter,
                                                np.logical_and(lag_filter, list_filter)))
    data_to_use = all_crps.loc[data_filter, :]
    af.e3fig(data_to_use, results_dir + "E3_crp_list2", which_list)
