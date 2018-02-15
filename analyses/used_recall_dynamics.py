import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import xarray as xr
# from numba import jit





def apply_to_zeros(lst, dtype=np.int64):
    """
    Convert a list of arrays to a 2d array padded with zeros to the right.
    """
    # determine the inner max length
    inner_max_len = max(map(len, lst))

    # allocate the return array
    result = np.zeros([len(lst), inner_max_len], dtype)

    # loop over the list and fill the non-zero entries
    for i, row in enumerate(lst):
        # fill the row
        result[i,:len(row)] = row
        #for j, val in enumerate(row):
        #    result[i][j] = val
    return result



def prec(outpos=1, listlen=None, recalls=None, filter_ind=None, **kwargs):
    """
    Calculate probability of recall as a function of output position.
    """
    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if isinstance(recalls,list):
        # convert to padded array
        recalls = apply_to_zeros(recalls)

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)


    # TODO: is this correct?? only tested for a single list, not a matrix!
        def nrecalled(row):
            no_repeats = np.unique(row)
            return sum(no_repeats > 0)

    return sum(np.apply_along_axis(nrecalled, axis=1, arr=recalls)) / float(listlen * recalls.shape[0])


def spc(listlen=None, recalls=None, filter_ind=None, **kwargs):
    """
    Calculate the serial position curve for a list of recall lists.
    """
    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if isinstance(recalls,list):
        # convert to padded array
        recalls = apply_to_zeros(recalls)

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)

    # loop over serial positions to get vals
    serpos = range(1,listlen+1)
    vals = [((recalls[filter_ind]==p).sum(1)>0).mean() for p in serpos]
    return np.rec.fromarrays([serpos,vals], names='serial_pos,prec')


def crp(listlen=None, recalls=None, filter_ind=None,
        allow_repeats=False, exclude_op=0, **kwargs):
    """
    Calculate a conditional response probability.

    Returns a recarray with lags, mcrp, ecrp, crpAll.
    """


    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if isinstance(recalls, list):
        # convert to padded array
        recalls = apply_to_zeros(recalls)

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)

    # determine possible lags
    lags = np.arange(0, 2 * listlen - 1) - (listlen - 1)

    # reset the numerator and denominator
    numer = np.zeros(len(lags), np.float64)
    denom = np.zeros(len(lags), np.float64)

    # loop over the lists
    for lis in recalls[filter_ind]:

        # loop over items in the list
        for r in np.arange(exclude_op, len(lis) - 1):
            # get the items
            i = lis[r]
            j = lis[r + 1]

            # see if increment, must be:
            # 1) positive serial positions (not intrusion)
            # 2) not immediate repetition
            # 3) not already recalled
            # 4) any optional conditional
            # if opt_cond is not None:
            #     opt_res = eval(opt_cond)
            # else:
            opt_res = True

            if (i > 0 and j > 0 and
                            i - j != 0 and
                    not np.any(np.in1d([i, j], lis[0:r])) and
                    opt_res):
                # not any(setmember1d([i,j],lis[0:r]))):

                # increment numerator
                lag = j - i
                nInd = np.nonzero(lags == lag)[0]
                numer[nInd] = numer[nInd] + 1

                # get all possible lags
                negLag = np.arange(i - 1) - (i - 1)
                posLag = np.arange(i, listlen) - (i - 1)
                allLag = np.union1d(negLag, posLag)

                # remove lags to previously recalled items
                if not allow_repeats:
                    recInd = np.nonzero(lis[0:r] > 0)[0]
                    recLag = lis[recInd] - i
                    goodInd = np.nonzero(~np.in1d(allLag, recLag))[0]
                    # goodInd = nonzero(~setmember1d(allLag,recLag))[0]
                    allLag = allLag[goodInd]

                # increment the denominator
                dInd = np.nonzero(np.in1d(lags, allLag))[0]
                # dInd = nonzero(setmember1d(lags,allLag))[0]
                denom[dInd] = denom[dInd] + 1

    # add in the subject's crp
    denom[denom == 0] = np.nan
    # numer[numer == 0] = 1
    crp_val = numer / denom
    # crp_val = crp_val - crp_val[~np.isnan(crp_val)].max()

    # return the values
    # add numer and denom to rec array and plot as a function of prec
    return np.rec.fromarrays([lags, crp_val, numer, denom], names='lag,crp,numer,denom')


def crp_graident(listlen=None, recalls=None, filter_ind=None,
        allow_repeats=False, exclude_op=0, **kwargs):

    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if isinstance(recalls, list):
        # convert to padded array
        recalls = apply_to_zeros(recalls)

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)

    # determine possible lags
    lags = np.arange(0, 2 * listlen - 1) - (listlen - 1)

    # reset the numerator and denominator
    numer = np.zeros(len(lags), np.float64)
    denom = np.zeros(len(lags), np.float64)

    # loop over the lists non-conditional rp
    for lis in recalls[filter_ind]:

        # loop over items in the list
        for r in np.arange(exclude_op, len(lis) - 1):
            # get the items
            i = lis[r]
            j = lis[r + 1]

            # increment numerator
            lag = j - i
            nInd = np.nonzero(lags == lag)[0]
            numer[nInd] = numer[nInd] + 1

            # increment the denominator
            denom = denom + 1

    # add in the subject's rp
    denom[denom == 0] = np.nan
    denom[lags == 0] = np.nan
    crp_val = numer / denom
    crp_val[crp_val == 0] = np.nan
    crp_val = np.rec.fromarrays([lags, crp_val, numer, denom], names='lag,crp,numer,denom')

    # first, compute the normal crp and make it an xarray
    # crp_val = crp(listlen, recalls, filter_ind, allow_repeats, exclude_op=exclude_op, **kwargs)
    crp_array = xr.DataArray(np.expand_dims(crp_val.crp, axis=0), coords=[("row", [0]), ("lag", crp_val.lag)]) # note we have to add an extra dimesnion to the crp_array, othewise lag is the only dimension and we cant eaisly add more rows for other subjects or lists or whatever
    # return  crp_array

    # compute the percent change from lag i to i+1
    max_lag = crp_array.coords['lag'].max()
    min_lag = crp_array.coords['lag'].min()
    lag_i = range(min_lag + 1, 0) + range(0, max_lag) # from |1| to |max_lag|-1
    lag_i_plus_1 = range(min_lag, -1) + range(1, max_lag + 1) # from |2| to |max_lag|
    return crp_array.sel(lag=lag_i) / crp_array.sel(lag=lag_i_plus_1).values


def tf_graident(listlen=None, recalls=None, filter_ind=None,
        allow_repeats=False, exclude_op=0, **kwargs):

    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if isinstance(recalls, list):
        # convert to padded array
        recalls = apply_to_zeros(recalls)

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)

    # the border between your near and far bins. Any lag less than or equal to bin_boarder will be counted as near
    bin_boarder = np.array([1])

    # reset the numerator and denominator
    numer = np.zeros(2, np.float64)
    denom = np.zeros(2, np.float64)

    # loop over the lists non-conditional rp
    for lis in recalls[filter_ind]:

        # loop over items in the list
        for r in np.arange(exclude_op, len(lis) - 1):
            # get the lag
            i = lis[r]
            j = lis[r + 1]
            lag = np.abs(j - i)

            # increment numerator and denominator for near vs far
            if lag <= bin_boarder:
                numer[0] = numer[0] + 1
                denom = denom + 1
            elif lag > bin_boarder:
                numer[1] = numer[1] + 1
                denom = denom + 1


            # lag_1 vs lag_2
            # if lag == 1:
            #     numer[0] = numer[0] + 1
            # elif lag == 2:
            #     numer[1] = numer[1] + 1

            # increment the denominator


    # add in the subject's rp
    denom[denom == 0] = np.nan
    numer[numer == 0] = np.nan
    crp_val = numer / denom

    # get ratio
    return crp_val[0] / crp_val[1]






def trans_fact(recs, dists):
    """
    Calculate transition factor.

    dists = -squareform(pdist(np.array([range(list_len)]).T))

    """

    # make sure recs are array
    recs = np.asanyarray(recs)

    # get lengths
    list_len = len(dists)
    nrecs = len(recs)

    # initialize containers
    tfs = np.empty(nrecs)*np.nan
    #weights = np.zeros(nrecs)

    # init poss ind
    poss_ind = np.arange(list_len)

    # loop over items
    for i in xrange(1,nrecs):
        # if current is 0, then stop
        if recs[i] == 0:
            break

        # make sure
        # 1) current and prev valid
        # 2) not a repeat
        if ((recs[i-1]>0) and (recs[i]>0) and
            (not recs[i] in recs[:i])):
            # get possible
            ind = poss_ind[~np.in1d(poss_ind, recs[:i]-1)]
            act_ind = poss_ind[ind] == (recs[i]-1)

            if (len(ind) == 1):
                # there are not any more possible recalls other than
                # this one so we're done
                continue

            # rank them
            ranks = rankdata(dists[int(recs[i-1])-1][ind])
            #print ranks

            # set the tf for that transition
            tfs[i] = (ranks[act_ind]-1.)/(len(ind)-1.)

            # fiddling with weights
            #weights[i] = (ranks[act_ind])/(2.*ranks[~act_ind].mean())
            #weights[i] = np.abs(ranks[act_ind] - ranks[~act_ind]).mean()/(ranks[act_ind] - ranks[~act_ind]).std()
            #weights[i] = ranks[act_ind]/(2.*ranks[~act_ind].mean())

    return tfs  # ,weights


def tem_fact(listlen=None, recalls=None, filter_ind=None, **kwargs):
    """
    """
    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if isinstance(recalls, list):
        # convert to padded array
        recalls = apply_to_zeros(recalls)

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)

    # get the dist factor
    dists = -squareform(pdist(np.array([range(listlen)]).T))

    # get pos and neg only
    #pos_dists = dists.copy()
    #pos_dists[np.tril_indices(listlen,1)] = np.nan
    #neg_dists = dists.copy()
    #neg_dists[np.triu_indices(listlen,1)] = np.nan

    # loop over the lists
    res = []
    for i, recs in enumerate(recalls[filter_ind]):
        # get the full tfact
        tfs = trans_fact(recs, dists)

        # get the direction
        rtemp = recs.copy().astype(np.float)
        rtemp[rtemp<=0] = np.nan
        lags = np.diff(rtemp)
        lags = np.insert(lags, 0, np.nan) # add a nan at begining to say that the first item did not have a transition
        # lags = np.array([np.nan] + lags.tolist()).astype(np.int) # dont understand why the original code did this, makes all the nans a number

        # append the recarray of results
        res.append(np.rec.fromarrays([[i+1]*len(tfs),recs[:len(tfs)],tfs,lags],
                                     names='list_num,rec_item,tf,lag'))
    return np.concatenate(res)


def permutation_loop(recalls, statistic_func, listlen, filter_ind):

        # permute the order of each list---not as simple as it seems because have to deal with nans each line must
        # have the actual recalls first and the nans last
        shuffled_recalls = np.empty(recalls.shape) * np.nan
        for row_i, lis in enumerate(recalls):
            new_order = np.random.permutation(lis[np.logical_not(np.isnan(lis))])
            shuffled_recalls[row_i, 0:new_order.shape[0]] = new_order

        # compute the statistic on the permuted data, and add it to the results matrix
        return statistic_func(listlen, shuffled_recalls, filter_ind)




def relative_to_random(listlen=None, recalls=None, filter_ind=None, statistic_func=None, data_col=None, n_perms=10000, par=True, POOL=None, **kwargs):

    if statistic_func is None or data_col is None:
        raise ValueError("You must pass a function to compute a statistic on your data, like prec or crp and you must"
                         "specify what column has the data you want to grab, like prec")

    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)

    # compute the statistic for the actual data
    out = statistic_func(listlen, recalls, filter_ind)
    results = out[data_col]


    if par:
        # num_cores = multiprocessing.cpu_count()
        # POOL = Parallel(n_jobs=num_cores, verbose=5)
        inputs = np.arange(0, n_perms)
        par_out = POOL(delayed(permutation_loop)(recalls, statistic_func, listlen, filter_ind) for i in inputs)
        for i in par_out:
            results = np.vstack((results, i[data_col]))
    # else:
    #     for pi in np.arange(0, n_perms):
    #
    #         # permute the order of each list---not as simple as it seems because have to deal with nans each line must
    #         # have the actual recalls first and the nans last
    #         shuffled_recalls = np.empty(recalls.shape) * np.nan
    #         for row_i, lis in enumerate(recalls):
    #             new_order = np.random.permutation(lis[np.logical_not(np.isnan(lis))])
    #             shuffled_recalls[row_i, 0:new_order.shape[0]] = new_order
    #
    #         # compute the statistic on the permuted data, and add it to the results matrix
    #         out = statistic_func(listlen, shuffled_recalls, filter_ind)
    #         results = np.vstack((results, out[data_col]))

    # compute zscores
    out[data_col] = (results[0, :] - np.nanmean(results[1:, :], axis=0)) / np.nanstd(results[1:, :], axis=0)
    # out[data_col] = results[0, :] / np.nanmean(results[1:, :], axis=0)
    return out





def el_flat_loop(listlen, recalls, filter_ind, allow_repeats, exclude_op, spc=None):

    # # #########option #1: recall the same number of items as were on the list, but not necessarily the same items, probability matching spc
    # # randomly recall n items for each list---not as simple as it seems because have to deal with nans each line must
    # # have the actual recalls first and the nans last
    # random_recalls = np.empty(recalls.shape) * np.nan
    # items = np.arange(0, listlen)
    # for row_i, lis in enumerate(recalls):
    #     num_to_recall = np.count_nonzero(
    #         np.unique(lis) > 0)  # exclude 1) blank cells (nan or zero), 2) repetitions, 3) intrusions
    #     recalled_items = np.random.choice(items, size=num_to_recall, replace=False, p=spc)
    #     # recall_order = np.random.permutation(recalled_items)
    #     random_recalls[row_i, 0:recalled_items.shape[0]] = recalled_items

    # #########option #2: recall the same number of items as were on the list, but not necessarily the same items
    # randomly recall n items for each list---not as simple as it seems because have to deal with nans each line must
    # have the actual recalls first and the nans last
    random_recalls = np.empty(recalls.shape) * np.nan
    items = np.arange(0, listlen)
    for row_i, lis in enumerate(recalls):
        num_to_recall = np.count_nonzero(np.unique(lis)>0)  # exclude 1) blank cells (nan or zero), 2) repetitions, 3) intrusions
        recalled_items = np.random.choice(items, size=num_to_recall, replace=False)
        # recall_order = np.random.permutation(recalled_items)
        random_recalls[row_i, 0:recalled_items.shape[0]] = recalled_items

    # #########option #3: just shuffle the actual recalls
    # # permute the order of each list---not as simple as it seems because have to deal with nans each line must
    # # have the actual recalls first and the nans last
    # random_recalls = np.empty(recalls.shape) * np.nan
    # for row_i, lis in enumerate(recalls):
    #     new_order = np.random.permutation(lis[np.logical_not(np.isnan(lis))])
    #     random_recalls[row_i, 0:new_order.shape[0]] = new_order

    # compute the statistic on the shuffled data
    flat_crp = crp(listlen, random_recalls, filter_ind, allow_repeats, exclude_op)

    # get the difference between observed and shuffled
    return flat_crp



def baseline_corrected_crp(listlen=None, recalls=None, filter_ind=None, allow_repeats=False, exclude_op=0, n_perms=1000, POOL=None, **kwargs):

    # todo: this is computing the full crp. But all we really need is the actual count. Should fix so that is all it does. Big overhead

    # compute the crp for the actual data
    observed_crp = crp(listlen, recalls, filter_ind, allow_repeats, exclude_op)

    # compute the SPC for the actual data and normalize to sum to one as we will need it to determine the probability
    spc_obs = spc(listlen=listlen, recalls=recalls, filter_ind=filter_ind).prec
    spc_obs = spc_obs / spc_obs.sum()

    # compute the statistic for many shuffles of the actual data. Using parallel processing
    # num_cores = multiprocessing.cpu_count()
    # pool = Parallel(n_jobs=num_cores, verbose=5)
    inputs = np.arange(0, n_perms)
    par_out = POOL(delayed(el_flat_loop)(listlen, recalls, filter_ind, allow_repeats, exclude_op, spc_obs) for i in inputs)
    results = observed_crp
    for i in par_out:
        results = np.vstack((results, i))

    theoretical_crp = np.nanmean(results['numer'][1:], axis=0)

    # single number summary
    observed_num_transitions = observed_crp.numer.sum()
    observed_sum_of_lags = np.sum(np.abs(observed_crp.lag)*observed_crp.numer)
    observed_avg_lag = observed_sum_of_lags / observed_num_transitions
    theoretical_num_transitions = np.sum(results['numer'][1:])
    theoretical_sum_of_lags = np.sum(results['numer'][1:] * np.abs(observed_crp.lag))
    theoretical_avg_lag = theoretical_sum_of_lags / theoretical_num_transitions

    theoretical_crp[theoretical_crp == 0] = np.nan  #  make any zeros nan to avoid division by 0 and inf values
    corrected_crp = pd.DataFrame.from_records(observed_crp)
    corrected_crp['corrected_crp'] = (observed_crp['numer'] - theoretical_crp) / theoretical_crp
    lag1 = corrected_crp.lag.abs() == 1
    lag35 = np.in1d(corrected_crp.lag.abs(), [3, 4, 5])
    corrected_crp['corrected_bias'] = corrected_crp.loc[lag1, 'corrected_crp'].mean() - corrected_crp.loc[lag35, 'corrected_crp'].mean()
    return corrected_crp


