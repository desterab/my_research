from anal_funcs import *

# load the data
all_crps, all_spcs = load_the_data()
all_crps.to_pickle(all_crps + ".pkl")
all_spcs.to_pickle(all_spcs + ".pkl")

# check sample sizes
print pd.crosstab(all_crps.lag, [all_crps.instruction_condition, all_crps.task_condition])

# make the figures
fig_compare_tasks(all_crps, all_spcs, which_cond=0, which_list=0, print_to='explicit_l0')
fig_compare_tasks(all_crps, all_spcs, which_cond=0, which_list=1, print_to='explicit_l1')
fig_compare_tasks(all_crps, all_spcs, which_cond=1, which_list=0, print_to='implicit_l0')
fig_compare_tasks(all_crps, all_spcs, which_cond=1, which_list=1, print_to='implicit_l1')

