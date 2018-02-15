import sys,os,os.path
from cbcc_tools.data_munging import preprocess_fr_exp_psiturk
import pandas as pd
import time
import os
import anal_funcs as af

# set paths
db_url = "mysql://khealey:Bib96?reply@127.0.0.1/Heal16implicit"  # url for the database in which raw psiturk ouput is stored
table_name = 'E1_item_relational_debug'  # table of the database
dict_path = "/home/khealey/code/experiments/Heal16implicit/static/js/word_pool/websters_dict.txt"  # dictionary to use when looking for ELIs and correcting spelling
save_file = 'HealEtal16implicit.data'
exp_data_dir = "/fmri2/PI/healey/data/HealEtal16implicit"  # path to data dir on circ2

# load the data from the psiturk experiment database and make it into a free recall object
data = preprocess_fr_exp_psiturk.load_psiturk_data(db_url, table_name)
data.to_pickle(save_file + ".raw" + ".pkl")




# recalls = af.make_psiturk_recall_matrix(data, dict_path)
#
# # save the data to a pickle file (once to a file that will become the current working data, and once to a dated
# # file that will be a backup
# recalls.to_pickle(save_file + ".pkl")
# recalls.to_pickle(time.strftime("%Y%m%d-%H%M%S") + save_file + ".pkl")
# data.to_pickle(time.strftime("%Y%m%d-%H%M%S") + save_file + ".raw" + ".pkl")
#
# # put copies on circ2
# os.system("scp " + "*" + save_file + "*" + " circ2.psy.msu.edu:" + exp_data_dir)
# os.system("scp " + time.strftime("%Y%m%d-%H%M%S") + save_file + " circ2.psy.msu.edu:" + exp_data_dir)