import numpy as np
from numba import jit
from cbcc_tools.beh_anal import recall_dynamics as cbcc

def evaluate(parameters, n_subjects, n_lists, n_items, data_vector):
    # run the model
    prec, temp_fact = tcm(parameters, n_subjects, n_lists, n_items)

    # compute rmsd
    model_vector = np.append(prec, temp_fact)
    return np.sqrt(np.nansum((data_vector - model_vector) ** 2) / data_vector.size)



# @jit(nopython=False, nogil=True, cache=True)
def tcm(parameters, n_subjects, n_lists, n_items):
    """
    TCM using a luce choice rule with a stopping graident
    :param parameters: vector of parameter values
    :param n_subjects: number of subjects to simulate
    :param n_lists:  number of lists per subject
    :param n_items: number of items per list
    :return: pfr, spc, and lag-crp matricies
    """
    ################## setup network
    # minimum activation of f_i during recall
    min_a = np.array(10.0**-7.)

    # passing the iterators as arguments is faster than making the model remake them with each eval
    sub_iter = range(n_subjects)
    list_iter = range(n_lists)
    item_iter = range(n_items)

    # the network has one more element than the number of list items --- for the pre-list context element
    n_elements = n_items + 1

    # initial context representation orthogonal to item representations
    # todo: decide if we should really be associating the first item with this pre-list element
    c_i = np.zeros(n_elements)
    c_i[-1] = 1
    c_i = c_i

    # item_representations
    item_representations = np.eye(n_elements)

    # create pre-experimental and experimental associative matrices. pre-experimental has 1 for
    # auto-associations, zero elsewhere. Experimental is all zeros (it updates as list is learned)
    m_fc_pre = np.eye(n_elements)
    m_cf_pre = np.eye(n_elements)
    m_fc_exp = np.zeros((n_elements, n_elements))
    m_cf_exp = np.zeros((n_elements, n_elements))

    ################## unpack parameters
    phi_s = parameters[0]
    phi_d = parameters[1]
    gamma_fc = parameters[2]
    gamma_cf = 0.8  # parameters[3]
    beta_enc = parameters[3]
    theta_s = parameters[4]
    theta_r = parameters[5]
    tau = parameters[6]
    beta_rec = parameters[7]

    # setup matricies to hold the recall dynamics results
    # todo: these could potentially be in network
    recalled_items = -np.ones((n_subjects, n_lists, n_items), dtype='int64')

    ##################  study the items
    for item_i in item_iter:
        ####### activate elements
        # activate features
        f_i = item_representations[item_i, :]

        ####### update associations
        # we are just tracking experimental changes like in sede08 and not adding to pre_exp until recall

        # get primacy gradient (note: exp should be phi_d \times j-1 but because of zero indexing j-1 = item_i)
        # todo:  should be able to precompute this for all run with a given param
        # value given it is just a function of item number perhaps a dictionary with autoupdate?
        phi_i = phi_s * np.exp(-phi_d * item_i) + 1

        # calculate the deltas
        delta_fc = np.outer(c_i.T, f_i)
        delta_cf = np.outer(f_i.T, c_i) * phi_i

        # update the experimental matrices
        # todo: see how much slower using += is throughout the whole model
        m_fc_exp += delta_fc
        m_cf_exp += delta_cf

        ##### update conntext
        c_i = evolve_context(c_i, f_i, m_fc_pre, beta_enc)
        # assert math.isclose(np.linalg.norm(c_i), 1.0, rel_tol=1e-5)

    # save the end-of-list state of context so we can reuse it for each subject and list during recall below
    c_i_end_list = c_i

    # combine pre-experimental and experimental matrices
    m_fc = (1 - gamma_fc) * m_fc_pre + gamma_fc * m_fc_exp
    m_cf = (1 - gamma_cf) * m_cf_pre + gamma_cf * m_cf_exp

    ##################  recall period

    # get stopping probability as a function of recall attempt according to equation:
    # P(stop, j) = \theta_s e^{j\theta_r}
    # we can do this here because it is constant for a given parameter set
    p_stop = theta_s * np.exp(np.arange(n_items) * theta_r)
    p_stop[p_stop > 1] = 1
    p_stop[p_stop < 0] = 0

    for subject_i in sub_iter:
        for list_i in list_iter:

            # set c_i to the saved end of list context and setup various counters
            c_i = c_i_end_list
            available_items = np.arange(n_items)

            for recall_attempt_i in item_iter:

                # see if we are going to halt recall by drawing from the binominal distribution with p = p_stop

                # use context to activate the feature layer
                f_i = np.dot(m_cf, c_i)
                f_i = f_i[:-1]  # to keep the pre-list element out of the competition

                # Find the probability of recalling each item according to luce choice rule:
                # p(i) = (1-P(stop))\frac{a^\tau_i}{\sum_k^Na^\tau_k}
                # then pick an item randomly from the resulting multinomial distribution
                a_i_tau = f_i**tau
                a_i_tau[a_i_tau < min_a] = min_a  # replace values that are effectively zero
                p_i = a_i_tau[available_items] / np.sum(a_i_tau[available_items]) * (1 - p_stop[recall_attempt_i])

                which_list_item = np.random.rand() < np.cumsum(p_i)
                if ~np.any(which_list_item):
                    break  # falls into the p_stop category --- we are done for this list.
                j = available_items[np.argmax(which_list_item)]
                recalled_items[subject_i, list_i, recall_attempt_i] = j

                # remove j from the available items
                available_items = available_items[available_items != j]

                # activate recall item on feature layer and update context
                f_i = item_representations[j, :]
                c_i = evolve_context(c_i, f_i, m_fc, beta_rec)
    return np.nanmean(cbcc.prec(recalled_items, n_items)), np.nanmean(cbcc.temporal_factor(recalled_items, n_items))


@jit(nopython=True, nogil=True, cache=True)
def evolve_context(c_i, f_i, matrix, beta):
    c_in = np.dot(matrix, f_i)
    c_in = c_in / np.sqrt(np.power(c_in, 2).sum(axis=0))  # normalize to ||c_i|| = 1
    c_dot = np.dot(c_i, c_in)
    rho = np.sqrt(1 + (beta ** 2) * ((c_dot ** 2) - 1)) - beta * c_dot
    return rho * c_i + beta * c_in



