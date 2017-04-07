import seaborn as sns
import anal_funcs as af
import numpy as np

# params for data prep and saving results
results_dir = "/Users/khealey/Library/Mobile Documents/com~apple~CloudDocs/lab/code/experiments/Heal16implicit/dissemination/manuscript/first_submission/figures/"
remake_data_file = False
n_perms = 1

# figure style params
sns.set_style("ticks")
sns.set_context("talk", font_scale=2.5, rc={"lines.linewidth": 4})
colors = ["#000000", "#808080", "#D3D3D3"]  # black and white
sns.set_palette(colors)

# load or create the data
all_crps = af.load_the_data(n_perms=n_perms, remake_data_file=remake_data_file, save_name=results_dir)

# make table 1
all_crps = af.sample_size_table(all_crps, results_dir)

# make Firgure 1
which_list = 0
data_filter = np.logical_and(all_crps.task_condition == "Shoebox", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
af.encoding_instruct_fig(data_to_use, which_list, results_dir + "Shoebox")

# make figure 2
which_list = 0
data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
af.encoding_instruct_fig(data_to_use, which_list, results_dir + "FrontDoor")

# make figure 3
which_instruction_cond = "Incidental"
which_list = 0
data_filter = np.logical_and(np.logical_and(all_crps.task_condition != "Shoebox", all_crps.task_condition != "Front Door"), all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
af.processing_task_fig(data_to_use, which_instruction_cond, which_list, results_dir + 'E3')
