# todo: sanity check for 1 list per participant
# todo: how to save the survey data to circ2 --- probably save both recalls and data.

import sys,os,os.path
sys.path.append("/home/khealey/code/py_modules/")
from beh_tools import psiturk_tools
import pandas as pd
import time
import os


# set paths
db_url = "mysql://khealey:Bib96?reply@127.0.0.1/for_debuging"  # url for the database in which raw psiturk ouput is stored
table_name = 'finalrunthrough'  # table of the database
dict_path = "/home/khealey/code/experiments/turkFR_template/static/js/word_pool/websters_dict.txt"  # dictionary to use when looking for ELIs and correcting spelling
save_file = 'turkFR_template.data.csv'
exp_data_dir = "/fmri2/PI/healey/data/turkFR_template"  # path to data dir on circ2

# load the data from the psiturk experiment database and make it into a free recall object
data = psiturk_tools.load_psiturk_data(db_url, table_name)
recalls = psiturk_tools.make_psiturk_recall_matrix(data, dict_path)

# save the data to a csv file (once to a file that will become the current working data, and once to a dated
# file that will be a backup
recalls.to_csv(save_file)
recalls.to_csv(time.strftime("%Y%m%d-%H%M%S") + save_file)

# put copies on circ2
os.system("scp " + save_file + " circ2.psy.msu.edu:" + exp_data_dir)
os.system("scp " + time.strftime("%Y%m%d-%H%M%S") + save_file + " circ2.psy.msu.edu:" + exp_data_dir)


