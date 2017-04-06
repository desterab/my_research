import os
import pickle
from anal_funcs import *
from joblib import Parallel
import multiprocessing



results_dir = "/Users/khealey/Library/Mobile Documents/com~apple~CloudDocs/lab/code/experiments/Heal16implicit/dissemination/manuscript/first_submission/figures/"



##################################################################################################
#  load and analyse data

remake_data_file = True

# load or create the data
if os.path.isfile("all_crps.pkl") and not remake_data_file:
    all_crps = pickle.load(open("all_crps.pkl", "rb"))
    all_spcs = pickle.load(open("all_spcs.pkl", "rb"))
else:
    num_cores = multiprocessing.cpu_count()
    with Parallel(n_jobs=num_cores, verbose=0) as POOL:
        all_crps, all_spcs = load_the_data(n_perms=10000, pool=POOL, save_name=results_dir + 'Heal16implicit_data.csv')
    all_crps.to_pickle("all_crps.pkl")
    all_spcs.to_pickle("all_spcs.pkl")

# take mean within subject and lag, check sample sizes
all_crps = all_crps.groupby(['subject', 'lag', 'list'], as_index=False).mean()
all_spcs = all_spcs.groupby(['subject', 'serial_pos', 'list'], as_index=False).mean()

# change conditions from numerical to verbal labelsls
all_crps.loc[all_crps.task_condition == 0, 'task_condition'] = "Shoebox"
all_crps.loc[all_crps.task_condition == 1, 'task_condition'] = "Movie"
all_crps.loc[all_crps.task_condition == 2, 'task_condition'] = "Relational"
all_crps.loc[all_crps.task_condition == 3, 'task_condition'] = "Scenario"
all_crps.loc[all_crps.task_condition == 4, 'task_condition'] = "Animacy"
all_crps.loc[all_crps.task_condition == 5, 'task_condition'] = "Weight"
all_crps.loc[all_crps.task_condition == 6, 'task_condition'] = "Front Door"
all_crps.loc[all_crps.instruction_condition == 0, 'instruction_condition'] = "Explicit"
all_crps.loc[all_crps.instruction_condition == 1, 'instruction_condition'] = "Incidental"






##################################################################################################
#  sample size table

# get overall sample sizes
n_tested = pd.crosstab(all_crps[np.logical_and(all_crps.lag==0,all_crps.list==0)].lag, [all_crps.instruction_condition, all_crps.task_condition])

# get number excluded for being aware
all_crps['aware_keep'] = all_crps.aware_check == 0
n_aware = pd.crosstab(all_crps[np.logical_and(all_crps.lag==0,all_crps.list==0)].aware_check, [all_crps.instruction_condition, all_crps.task_condition])


# get number excluded for poor recall
all_crps['prec_keep'] = all_crps.prec > 0
n_Prec = pd.crosstab(all_crps[np.logical_and(all_crps.lag==0,all_crps.list==0)].prec_keep, [all_crps.instruction_condition, all_crps.task_condition])

# exclude the bad people
data_filter = np.logical_and(all_crps.aware_keep, all_crps.prec_keep)
all_crps = all_crps.loc[ data_filter, :]

# final analysed sample size
n_included = pd.crosstab(all_crps[np.logical_and(all_crps.lag==0,all_crps.list==0)].lag, [all_crps.instruction_condition, all_crps.task_condition])

# compute average prec
prec_table = all_crps.loc[np.logical_and(all_crps.lag==0,all_crps.list==0), :]['prec'].groupby([all_crps.instruction_condition, all_crps.task_condition]).describe()

with open(results_dir + "table_values.tex", "w") as text_file:
    text_file.write('\\newcommand\\shoeExplicit{%s}\n' % n_tested['Explicit']['Shoebox'][0])
    text_file.write('\\newcommand\\shoeIncidental{%s}\n' % n_tested['Incidental']['Shoebox'][0])
    text_file.write('\\newcommand\\doorExplicit{%s}\n' % n_tested['Explicit']['Front Door'][0])
    text_file.write('\\newcommand\\doorIncidental{%s}\n' % n_tested['Incidental']['Front Door'][0])
    text_file.write('\\newcommand\\Movie{%s}\n' % n_tested['Incidental']['Movie'][0])
    text_file.write('\\newcommand\\Relational{%s}\n' % n_tested['Incidental']['Relational'][0])
    text_file.write('\\newcommand\\Scenario{%s}\n' % n_tested['Incidental']['Scenario'][0])
    text_file.write('\\newcommand\\Animacy{%s}\n' % n_tested['Incidental']['Animacy'][0])
    text_file.write('\\newcommand\\Weight{%s}\n' % n_tested['Incidental']['Weight'][0])

    text_file.write('\\newcommand\\shoeExplicitAware{--}\n' % n_aware['Explicit']['Shoebox'][1])
    text_file.write('\\newcommand\\shoeIncidentalAware{%s}\n' % n_aware['Incidental']['Shoebox'][1])
    text_file.write('\\newcommand\\doorExplicitAware{--}\n' % n_aware['Explicit']['Front Door'][1])
    text_file.write('\\newcommand\\doorIncidentalAware{%s}\n' % n_aware['Incidental']['Front Door'][1])
    text_file.write('\\newcommand\\MovieAware{%s}\n' % n_aware['Incidental']['Movie'][1])
    text_file.write('\\newcommand\\RelationalAware{%s}\n' % n_aware['Incidental']['Relational'][1])
    text_file.write('\\newcommand\\ScenarioAware{%s}\n' % n_aware['Incidental']['Scenario'][1])
    text_file.write('\\newcommand\\AnimacyAware{%s}\n' % n_aware['Incidental']['Animacy'][1])
    text_file.write('\\newcommand\\WeightAware{%s}\n' % n_aware['Incidental']['Weight'][1])

    # # commented out because there are no people with zero prec and the 1st entry thus does not exist and throws an error
    # text_file.write('\\newcommand\\shoeExplicitPrec{%s}\n' % n_Prec['Explicit']['Shoebox'][1])
    # text_file.write('\\newcommand\\shoeIncidentalPrec{%s}\n' % n_Prec['Incidental']['Shoebox'][1])
    # text_file.write('\\newcommand\\doorExplicitPrec{%s}\n' % n_Prec['Explicit']['Front Door'][1])
    # text_file.write('\\newcommand\\doorIncidentalPrec{%s}\n' % n_Prec['Incidental']['Front Door'][1])
    # text_file.write('\\newcommand\\MoviePrec{%s}\n' % n_Prec['Incidental']['Movie'][1])
    # text_file.write('\\newcommand\\RelationalPrec{%s}\n' % n_Prec['Incidental']['Relational'][1])
    # text_file.write('\\newcommand\\ScenarioPrec{%s}\n' % n_Prec['Incidental']['Scenario'][1])
    # text_file.write('\\newcommand\\AnimacyPrec{%s}\n' % n_Prec['Incidental']['Animacy'][1])
    # text_file.write('\\newcommand\\WeightPrec{%s}\n' % n_Prec['Incidental']['Weight'][1])


    text_file.write('\\newcommand\\shoeExplicitIncluded{%s}\n' % n_included['Explicit']['Shoebox'][0])
    text_file.write('\\newcommand\\shoeIncidentalIncluded{%s}\n' % n_included['Incidental']['Shoebox'][0])
    text_file.write('\\newcommand\\doorExplicitIncluded{%s}\n' % n_included['Explicit']['Front Door'][0])
    text_file.write('\\newcommand\\doorIncidentalIncluded{%s}\n' % n_included['Incidental']['Front Door'][0])
    text_file.write('\\newcommand\\MovieIncluded{%s}\n' % n_included['Incidental']['Movie'][0])
    text_file.write('\\newcommand\\RelationalIncluded{%s}\n' % n_included['Incidental']['Relational'][0])
    text_file.write('\\newcommand\\ScenarioIncluded{%s}\n' % n_included['Incidental']['Scenario'][0])
    text_file.write('\\newcommand\\AnimacyIncluded{%s}\n' % n_included['Incidental']['Animacy'][0])
    text_file.write('\\newcommand\\WeightIncluded{%s}\n' % n_included['Incidental']['Weight'][0])

    text_file.write('\\newcommand\\shoeExplicitPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Explicit"]["Shoebox"]["mean"], prec_table["Explicit"]["Shoebox"]["std"]))
    text_file.write('\\newcommand\\shoeIncidentalPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Shoebox"]["mean"], prec_table["Incidental"]["Shoebox"]["std"]))
    text_file.write('\\newcommand\\doorExplicitPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Explicit"]["Front Door"]["mean"], prec_table["Explicit"]["Front Door"]["std"]))
    text_file.write('\\newcommand\\doorIncidentalPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Front Door"]["mean"], prec_table["Incidental"]["Front Door"]["std"]))

    text_file.write(
        '\\newcommand\\MoviePrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Movie"]["mean"],
                                                                       prec_table["Incidental"]["Movie"]["std"]))
    text_file.write(
        '\\newcommand\\RelationalPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Relational"]["mean"],
                                                                       prec_table["Incidental"]["Relational"]["std"]))
    text_file.write(
        '\\newcommand\\ScenarioPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Scenario"]["mean"],
                                                                       prec_table["Incidental"]["Scenario"]["std"]))
    text_file.write(
        '\\newcommand\\AnimacyPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Animacy"]["mean"],
                                                                       prec_table["Incidental"]["Animacy"]["std"]))
    text_file.write(
        '\\newcommand\\WeightPrec{{{:.2f} ({:.2f})}}\n'.format(prec_table["Incidental"]["Weight"]["mean"],
                                                                       prec_table["Incidental"]["Weight"]["std"]))







##################################################################################################
#  Figures

# overall figure style
sns.set_style("ticks")
sns.set_context("talk", font_scale=2.5, rc={"lines.linewidth": 4})
colors = ["#000000", "#808080", "#D3D3D3"]  # black and white
sns.set_palette(colors)

##### shoebox
which_list = 0
data_filter = np.logical_and(all_crps.task_condition == "Shoebox", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
encoding_instruct_fig(data_to_use, which_list, results_dir + "Shoebox")

##### door
which_list = 0
data_filter = np.logical_and(all_crps.task_condition == "Front Door", all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
encoding_instruct_fig(data_to_use, which_list, results_dir + "FrontDoor")

##### many tasks
which_instruction_cond = "Incidental"
which_list = 0
data_filter = np.logical_and(np.logical_and(all_crps.task_condition != "Shoebox", all_crps.task_condition != "Front Door"), all_crps.lag.abs() <= 5)
data_to_use = all_crps.loc[data_filter, :]
processing_task_fig(data_to_use, which_instruction_cond, which_list, results_dir + 'E3')
