import numpy as np


def random_transition(params):

    # normalized, mean centered spc weights
    spc_weights = np.array(params['spc'])
    # spc_weights = spc_weights / spc_weights.sum()
    spc_weights = spc_weights - spc_weights.mean()

    # task parameters
    n_lists = params['n_lists']
    n_items = params['n_items']
    p_rec = params['p_rec']
    a_pos = params['a_pos']
    a_neg = params['a_neg']
    asym = params['asym']
    model_contiguity = params['model_contiguity']

    # preallocate some variables
    items = np.array(range(n_items))
    lists = range(n_lists)
    recalls = np.empty([n_lists, n_items]) * np.nan

    # recall prob for each serial position, making sure between 1 and 0
    recall_probs = p_rec + spc_weights
    recall_probs[recall_probs > 1.] = 1.
    recall_probs[recall_probs < 0.] = 0.

    # for each item on each list, see if it is recalled by drawing for a binomial
    these_positions = np.random.binomial(1, p=np.tile(recall_probs, (n_lists, 1)))

    # transition probabilities as a function of lag
    pos_lag_probs = asym * (np.arange(n_items-1)+1) ** -a_pos
    neg_lag_probs = (np.arange(n_items-1)+1) ** -a_neg
    lag_probs = np.concatenate((neg_lag_probs[::-1], pos_lag_probs))
    lags = np.concatenate((np.arange(-n_items+1, 0), np.arange(1, n_items)))

    # loop over lists and order the recalls
    if not model_contiguity:
        for list_i in lists:
            these_items = np.random.permutation(items[these_positions[list_i, :] == 1])
            recalls[list_i, 0:these_items.shape[0]] = these_items
    else:
        for list_i in lists:
            these_items = items[these_positions[list_i, :] == 1]
            for output, item in enumerate(these_items):

                # if this is the first output, grab a random item, otherwise grab based on prob dist across lags
                if output == 0:
                    recalled_index = np.random.randint(0, these_items.shape[0])
                else:
                    pos_lags = these_items - recalled_item
                    indicies_to_pos_lags = np.array([np.any([lag_i == pos_i for pos_i in pos_lags]) for lag_i in lags])
                    cur_probs = lag_probs[indicies_to_pos_lags]
                    cur_probs = cur_probs / cur_probs.sum()
                    actual_lag = np.random.choice(a=pos_lags, p=cur_probs)
                    recalled_index = np.where(pos_lags == actual_lag)[0]

                # put the recalled item in the recalls matrix and then remove it from these_items
                recalled_item = these_items[recalled_index]
                recalls[list_i, output] = recalled_item
                these_items = np.delete(these_items, recalled_index)

    # convert from zero indexing so recalls gives serial positions starting at one
    return recalls + 1






def tcm_lite(params):

    # task parameters
    n_lists = params['n_lists']
    n_items = params['n_items']
    n_recall_attempts = params['n_recall_attempts']

    # model parameters
    b_enc = params['b_enc']
    b_rec = params['b_rec']
    p_rec = params['p_rec']

    # preallocate some variables
    items = range(n_items)
    lists = range(n_lists)
    attempts = range(n_recall_attempts)
    recalls = np.empty([n_lists, n_recall_attempts]) * np.nan

    # loop over lists
    for list in lists:

        # initalize context and item vectors, and the associative matricies
        c = np.zeros(n_items)
        f = np.zeros(n_items)
        mfc = np.eye(n_items)
        mcf = np.eye(n_items)

        # present each item
        for i in items:

            # set f(i) to 1 and all other elements to zero
            f[i] = 1.
            f[:i] = 0.

            # compute input to context
            c_in = f.dot(mfc)

            # update context
            rho = np.sqrt(1 + (b_enc**2) * ((c.dot(c_in)**2) - 1)) - b_enc * c.dot(c_in) #TODO: make sure this agrees with matlab
            c = (rho * c) + (b_enc * c_in)

            # determine if we will successfully encode this item
            if np.random.binomial(1, p=p_rec) == 0:
                continue

            # update associations
            delta = np.outer(c.T, f)
            mcf = mcf + delta
            delta = np.outer(f.T, c)
            mfc = mfc + delta

        # recall some items
        output_pos = 0
        for attempt in attempts:

            # allow current context to activate features
            f = c.dot(mcf)

            # pick an item to recall
            f_sup = f / f.sum()
            sp = np.random.choice(a=items, p=f_sup)

            # move on if it is a repeat
            cur_recalls = recalls[list, :]
            repeats = [np.sum(a == sp) for a in cur_recalls]
            if np.sum(repeats) == 0:


                # record the recall
                recalls[list, output_pos] = sp

                # activate the recalled item on f
                f = np.zeros(n_items)
                f[sp] = 1

                # allow f to update context
                c_in = f.dot(mfc)
                rho = np.sqrt(1 + (b_rec**2) * ((c.dot(c_in)**2) - 1)) - b_rec * c.dot(c_in) #TODO: make sure this agrees with matlab
                c = (rho * c) + (b_rec * c_in)

                # update output position
                output_pos = output_pos + 1

    # convert from zero indexing so recalls gives serial positions starting at one
    return recalls + 1
