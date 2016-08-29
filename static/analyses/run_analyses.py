import sys,os,os.path
sys.path.append("/home/khealey/code/py_modules/")
from beh_tools import psiturk_tools
from anal_funcs import *
import pandas as pd

# load the data from the psiturk experiment database and make it into a free recall object
db_url = "mysql://khealey:Bib96?reply@127.0.0.1/for_debuging"
table_name = 'FRresults'
data = psiturk_tools.load_psiturk_data(db_url, table_name)
recalls = psiturk_tools.make_psiturk_recall_matrix(data)

# compute a SPC
# from beh_tools import recall_dynamics as rdf
# spc = rdf.spc()





