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
    data = data.loc[np.logical_or(~data['aware'], data['instruction_condition'] == 'Explicit'), :]
    sample_sizes_included_counts = pd.crosstab(data.aware, [data.instruction_condition, data.task_condition])


    # condition vector
    coords = {
        'subject': data.subject,
    }
    instruction_condition = xr.DataArray(data.instruction_condition, dims=('subject'), coords=coords)
    task_condition = xr.DataArray(data.task_condition, dims=('subject'), coords=coords)
    recall_instruction_condition = xr.DataArray(data.recall_instruction_condition, dims=('subject'), coords=coords)


    n_outputs = 19
    coords.update({'output_position': range(n_outputs+1)})
    coords.update({'list': [0]})
    rec_mat = data[data.columns[0:n_outputs+1]].values[:, np.newaxis, :]  # note adding an new axis for list

    # translate from beh_toolbox coding to cbcc_tools coding
    rec_mat = rec_mat - 1  # convert to zero indexing
    rec_mat[rec_mat == -1-1] = -2  # PLIs
    rec_mat[rec_mat == -999.0-1] = -3  # PLIs
    rec_mat[rec_mat == -1999.0-1] = -4  # nans in middle of sequence
    rec_mat[rec_mat == -2999.0-1] = -4  # non alpha string
    rec_mat[np.isnan(rec_mat)] = -1  # post last recall


    rec_mat = rec_mat.astype('int')
    recalls = xr.DataArray(rec_mat,
                           dims=('subject', 'list', 'output_position'), coords=coords)
    ds = xr.Dataset({
        'recalls': recalls,
        'instruction_condition': instruction_condition,
        'task_condition': task_condition,
        'recall_instruction_condition': recall_instruction_condition
    })
    ds.attrs['n_items_per_list'] = 16
    return ds, sample_sizes_aware_counts, sample_sizes_included_counts


def lag_crp_plot(lag_crp, n_lags_to_plot=5, error_type='CI'):
    # find indices to the lags we want to plot
    middle = int(np.floor(lag_crp.shape[1] / 2))
    these_lags = list(range(middle - n_lags_to_plot, middle + n_lags_to_plot + 1))

    # get mean
    # todo: this will return an empty slice warning because the lag zero col is always nan... could suppress warning
    m = np.nanmean(lag_crp[:, these_lags], axis=0)

    # get the requested error bar
    if error_type == 'boot_ci':
        # todo add 95% bootstraped CI and SEM
        print('sooo sad...')
        return
    else:  # everything else requires knowing SD
        e = np.nanstd(lag_crp[:, these_lags], ddof=1,
                      axis=0)  # ddof makes it divide SS/n-1 instead of SS/n, which as we all know is the right thing to do.
        if not error_type == 'sd':  # everything else require knowing SEM
            e = e / np.sqrt(lag_crp.shape[0])
            if not error_type == 'sem':  # must be CI
                e = e * 1.96

    # make and beautify plot
    plt.errorbar(x=np.arange(-n_lags_to_plot, n_lags_to_plot + 1), y=m, yerr=e)
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Prob.')
    plt.xticks(np.arange(-n_lags_to_plot, n_lags_to_plot + 1, 2))




sub1_data = pd.DataFrame.from_csv('/Users/khealey/code/experiments/Heal16implicit/dissemination/manuscript/jml/first_submission/figures/Heal16implicit_data.csv')
sub2_data = pd.DataFrame.from_csv('/Users/khealey/code/experiments/Heal16implicit/dissemination/manuscript/jml/second_submission/figures/Heal16implicit_data.csv')
all_data = pd.concat([sub1_data, sub2_data])

# split into lists
ds, sample_sizes_aware_counts, sample_sizes_included_counts = make_xarray(sub2_data)

# ds, sample_sizes_aware_counts, sample_sizes_included_counts = make_xarray(all_data.loc[all_data.list==0, :])



rdf.run_these_analyses(ds, ['pfr', 'spc', 'lag_crp'])


# load figure style sheet
plt.style.use('~/code/py_modules/cbcc_tools/plotting/stylesheets/cbcc.mplstyle')


# e1 crp for comparison
e1_explicit_filter = np.logical_and(ds.instruction_condition == 'Explicit', ds.task_condition == 'Shoebox')
e1_implicit_filter = np.logical_and(ds.instruction_condition == 'Incidental', ds.task_condition == 'Shoebox')
lag_crp_plot(ds.lag_crp[e1_explicit_filter])
lag_crp_plot(ds.lag_crp[e1_implicit_filter])
plt.ylim(0, .2)
plt.savefig('e1.pdf')
plt.close()


# e1 crp for comparison
e2_explicit_filter = np.logical_and(ds.instruction_condition == 'Explicit', ds.task_condition == 'Front Door')
e2_implicit_filter = np.logical_and(ds.instruction_condition == 'Incidental', ds.task_condition == 'Front Door')
# lag_crp_plot(ds.lag_crp[e2_explicit_filter])
lag_crp_plot(ds.lag_crp[e2_implicit_filter])
plt.ylim(0, .2)
plt.savefig('e2.pdf')
plt.close()



# e1 crp for comparison
e3_implicit_filter = np.logical_and(ds.instruction_condition == 'Incidental', ds.task_condition == 'Relational')
# lag_crp_plot(ds.lag_crp[e2_explicit_filter])
lag_crp_plot(ds.lag_crp[e3_implicit_filter])
plt.ylim(0, .15)
plt.savefig('e3.pdf')
plt.close()




# e1 crp for comparison
e4_implicit_filter = np.logical_and(np.logical_and(ds.instruction_condition == 'Incidental', ds.task_condition == 'Varying Size'), ds.recall_instruction_condition == 'Free')
# lag_crp_plot(ds.lag_crp[e2_explicit_filter])
lag_crp_plot(ds.lag_crp[e4_implicit_filter])
plt.ylim(0, .2)
plt.savefig('e4.pdf')
plt.close()

ds






