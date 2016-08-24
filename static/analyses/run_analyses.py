from beh_tools import psiturk_tools
from beh_tools import recall_dynamics as rdf
from anal_funcs import *
import pandas as pd

# load the data from the psiturk experiment database and make it into a free recall object
db_url = "mysql://khealey:Bib96?reply@127.0.0.1:3306/fr_turk_debug"
table_name = 'FRresults'
data = psiturk_tools.load_psiturk_data(db_url, table_name)
recalls = psiturk_tools.make_psiturk_recall_matrix(data)

# compute a SPC
spc = rdf.spc()





