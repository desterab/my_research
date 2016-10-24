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

print pd.crosstab(all_crps[np.logical_and(all_crps.lag==0,all_crps.list==0)].lag, [all_crps.instruction_condition, all_crps.task_condition])

# make the figures
apply_perm_correction = False
if apply_perm_correction:
    fig_prefix = "_perm"
else:
    fig_prefix = "_standard"
fig_compare_tasks(all_crps, all_spcs, which_cond=0, which_list=0, apply_perm_correction=apply_perm_correction, print_to="explicit" + fig_prefix + "_l0")
fig_compare_tasks(all_crps, all_spcs, which_cond=0, which_list=1, apply_perm_correction=apply_perm_correction, print_to='explicit' + fig_prefix + '_l1')
fig_compare_tasks(all_crps, all_spcs, which_cond=1, which_list=0, apply_perm_correction=apply_perm_correction, print_to='implicit' + fig_prefix + '_l0')
fig_compare_tasks(all_crps, all_spcs, which_cond=1, which_list=1, apply_perm_correction=apply_perm_correction, print_to='implicit' + fig_prefix + '_l1')

