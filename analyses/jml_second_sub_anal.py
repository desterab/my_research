import pandas as pd
import xarray as xr
import numpy as np
from cbcc_tools.beh_anal import recall_dynamics as rdf
from cbcc_tools.plotting import plotting_func as rdf_plot
import matplotlib.pyplot as plt
from matplotlib import gridspec

def make_xarray(data):

    # get sample sizes (aware and not) then get rid of aware people in the incidential conditons
    data['aware'] =data['aware'].str.contains('yes')
    sample_sizes_aware_counts = pd.crosstab(data.aware, [data.instruction_condition, data.task_condition])
    data = data.loc[np.logical_or(~data['aware'],data['instruction_condition'] == 'Explicit'), :]
    sample_sizes_included_counts = pd.crosstab(data.aware, [data.instruction_condition, data.task_condition])


    # condition vector
    coords = {
        'subject': data.subject,
    }
    instruction_condition = xr.DataArray(data.instruction_condition, dims=('subject'), coords=coords)
    task_condition = xr.DataArray(data.task_condition, dims=('subject'), coords=coords)
    recall_instruction_condition = xr.DataArray(data.recall_instruction_condition, dims=('subject'), coords=coords)


    coords.update({'output_position': range(25)})
    coords.update({'list': [0]})
    rec_mat = data[data.columns[0:25]].values[:, np.newaxis, :]  # note adding an new axis for list
    rec_mat = rec_mat - 1  # convert to zero indexing
    rec_mat[np.isnan(rec_mat)] = -1
    rec_mat = rec_mat.astype('int')
    recalls = xr.DataArray(rec_mat,
                           dims=('subject', 'list', 'output_position'), coords=coords)
    ds = xr.Dataset({
        'recalls': recalls,
        'instruction_condition': instruction_condition,
        'task_condition': task_condition
    })
    ds.attrs['n_items_per_list'] = 16
    return ds, sample_sizes_aware_counts, sample_sizes_included_counts



sub1_data = pd.DataFrame.from_csv('/Users/khealey/code/experiments/Heal16implicit/dissemination/manuscript/jml/first_submission/figures/Heal16implicit_data.csv')
sub2_data = pd.DataFrame.from_csv('/Users/khealey/code/experiments/Heal16implicit/dissemination/manuscript/jml/second_submission/figures/Heal16implicit_data.csv')
all_data = pd.concat([sub1_data, sub2_data])

##### assem
# ble xarray

# split into lists
ds, sample_sizes_aware_counts, sample_sizes_included_counts = make_xarray(all_data.loc[all_data.list==0, :])

rdf.run_these_analyses(ds, ['pfr', 'spc', 'crp'])

# setup the grid
fig = plt.figure(figsize=(30, 10))
gs = gridspec.GridSpec(1, 2)
crp_axis = fig.add_subplot(gs[0, 0])
rdf_plot.plot_spc(ds, crp_axis, hue='instruction_condition', condition_list=['instruction_condition'])




ds





