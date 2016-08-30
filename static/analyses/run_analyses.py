from beh_tools import recall_dynamics as rdf
import os
import pandas as pd


# paths
save_file = 'turkFR_template.data.csv'
exp_data_dir = "/fmri2/PI/healey/data/turkFR_template/"  # path to data dir on circ2

# load the pdanas dataframe from circ2
os.system("scp circ2.psy.msu.edu:" + exp_data_dir + save_file + " .")
recalls = pd.read_csv(save_file)

# compute a SPC
spc = rdf.spc()





