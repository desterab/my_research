#todo: allow cognates: e.g., for boy boys is good
#todo: sanity check to make sure recal is never more than 1
import numpy as np
from pyxdameraulevenshtein import damerau_levenshtein_distance_ndarray


def which_item(recall, presented, dictionary):
    """

    :param recall: the string typed in by the subject
    :param presented: a list of words seen by this subject so far in the experiment
    :param dictionary: a list of strings we want to consider as possible intrusions
    :return:
    """

    # does the recall exactly match a word that has been presented
    seen, seen_where = self_term_search(recall.response, presented.word)
    if seen:

        # could be a PLI
        intrusion = presented.iloc[seen_where].list != recall.list
        if intrusion:
            return -1 #todo: what are the standard matlab codes for ELI and PLI?

        # its not a PLI, so find the serial pos
        first_item = next(item for item, listnum in enumerate(presented.list) if listnum == recall.list)
        serial_pos = seen_where - first_item + 1
        return serial_pos

    # does the recall exactly match a word in the dictionary
    in_dict, where_in_dict = self_term_search(recall.response, dictionary)
    if in_dict:
        return -999 #todo: what are the standard matlab codes for ELI and PLI?

    # the closest match based on edit distance
    recall = correct_spelling(recall, presented, dictionary)
    return which_item(recall, presented, dictionary)


def self_term_search(find_this, in_this):
    for index, word in enumerate(in_this):
        if word == find_this:
            return True, index
    return False, None


def correct_spelling(recall, presented, dictionary):

    # edit distance to each item in the pool and dictionary
    dist_to_pool = damerau_levenshtein_distance_ndarray(recall.response, np.array(presented.word))
    dist_to_dict = damerau_levenshtein_distance_ndarray(recall.response, np.array(dictionary))

    # position in distribution of dist_to_dict
    ptile = np.true_divide(sum(dist_to_dict <= np.amin(dist_to_pool)), dist_to_dict.size)

    # decide if it is a word in the pool or an ELI
    if ptile <= .1: #todo: make this a param
        corrected_recall = presented.iloc[np.argmin(dist_to_pool)].word
    else:
        corrected_recall = dictionary[np.argmin(dist_to_dict)]
    recall.response = corrected_recall
    return recall


def make_recall_matrix(data):
    recalls = []

    # load the webesters dict
    dict_file = open("/Users/khealey/code/experiments/psiturk_exps/turk_fr/static/js/word_pool/websters_dict.txt", "r") #TODO: put this path as a param somewhere!
    dictionary = dict_file.read().split()
    dict_file.close()

    # first normalize the recalls, pools, and dictionary to get rid of obvious typos like caps and spaces and to make
    # everything lowercase
    data.word = data.word.str.strip().str.lower()
    data.response = data.response.str.strip().str.lower()
    dictionary = [word.lower().strip() for word in dictionary]

    # loop over subjects, for each isolate their data
    subjects = data.uniqueid.unique()
    for s in subjects:
        s_filter = data.uniqueid == s
        recalls_filter = data.phase == 'recall'
        study_filter = data.phase == 'study'
        cur_recalls = data.loc[s_filter & recalls_filter, ['list', 'response']]
        cur_items = data.loc[s_filter & study_filter, ['list', 'word']]

        # loop over this subjects recalls and for each find its serial position or mark as intrusion
        for index, recall in cur_recalls.iterrows():
            recall.response.strip().lower()
            which_item(recall, cur_items.loc[cur_items.list <= recall.list], dictionary)
    return recalls





