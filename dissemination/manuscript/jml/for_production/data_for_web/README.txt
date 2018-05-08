This archive contains a csv data with the data from the experiments reported in:

Healey, M. K. (2018). Temporal contiguity in incidentally encoded memories. Journal of Memory and Language. 

Refer to this manuscript for the methods of the experiments. 

Note that the paper reports 4 experiments. All of these are in a single data file with condition identifiers. 


This file written by Karl Healey (khealey@msu.edu). Send word if you find anything weird or out of sorts with the data!

%%%%%%%%%%%%%%%%%%%

If you open the file Heal18.csv you will find a number of columns.

The columns called output 1 through output 25, correspond to output positions in the recall sequence. The remaining columns give information like subject number, demographic information, encoding task and recall task (again see the paper for details on the experiments).

Each row represents 1 free recall list. Some subjects will have more than one row because the completed more than one list.

For a given list, the output columns give the serial positions of the recalled item.

Recalls are coded by serial position. For example, if output position 3 for a given list is coded as "2", it means the third word they recalled (output position 3) was in the 2nd serial poisition in the study list. 

Sometimes people recall things that were not on the list and therefore don't have serial position. The following codes are used to identify these recalls:

-1.0 = Prior list intrusion (recalling an item from an earlier list)
-999.0 = Extra list intrusion (recalling a real English word that was never presented to the subject)
-1999.0 = subject submitted a blank text-box without typing any word
-2999.0 = subject's response could not be scored because it contained non-letter characters (e.g., numbers, symbols)
