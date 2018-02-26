import anal_funcs as af
import numpy as np
import pickle

# params for data prep and saving results
results_dir = "../dissemination/manuscript/jml/second_submission/figures/"
dict_path = "/Users/khealey/code/py_modules/cbcc_tools/wordpool_files/websters_dict.txt"  # dictionary to use when looking for ELIs and correcting spelling
remake_data_file = False
n_perms = 10000

# load or create the recalls matrix
data = pickle.load( open( "HealEtal16implicit.data.raw.pkl", "rb" ) )
recalls = af.make_psiturk_recall_matrix(data, True, dict_path, 'HealEtal16implicit.recalls')

# load or compute the recall dynamics
all_crps = af.load_the_data(n_perms=n_perms, remake_data_file=True,
                            recalls_file='HealEtal16implicit.recalls.pkl', save_name=results_dir)

# all_spcs = af.get_spc(n_perms=n_perms, remake_data_file=True,
#                             recalls_file='HealEtal16implicit.recalls.pkl', save_name=results_dir)

# make table 1
all_crps = af.E4_sample_size_table(all_crps, results_dir)


# data_filter = np.logical_and(np.logical_or(all_crps.task_condition == "Constant Size",
#                                        all_crps.task_condition == "Varying Size"),
#                                     all_crps.lag.abs() <= 5)


which_list = 0
data_filter = np.logical_and(all_crps.task_condition == "Constant Size", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
af.E4_fig(data_to_use, which_list, results_dir + "E4_constant")

which_list = 0
data_filter = np.logical_and(all_crps.task_condition == "Varying Size", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
af.E4_fig(data_to_use, which_list, results_dir + "E4_varying")









### figures from original submission

# # make table 1
# all_crps = af.sample_size_table(all_crps, results_dir)

# # make Firgure 1
# which_list = 0
# data_filter = np.logical_and(all_crps.task_condition == "Shoebox", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# af.encoding_instruct_fig(data_to_use, which_list, results_dir + "Shoebox")
#
# # make figure 2
# which_list = 0
# data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# af.encoding_instruct_fig(data_to_use, which_list, results_dir + "FrontDoor")
#
# # make figure 3
# which_instruction_cond = "Incidental"
# which_list = 0
# data_filter = np.logical_and(np.logical_and(all_crps.task_condition != "Shoebox",
#                                             all_crps.task_condition != "Front Door"), all_crps.lag.abs() <= 5)
# data_to_use = all_crps.loc[data_filter, :]
# af.processing_task_fig(data_to_use, which_instruction_cond, which_list, results_dir + 'E3')
