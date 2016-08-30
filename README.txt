# to setup a new mySQL database for a new experiment

-start mySQL and create a new database
mysql -uroot -p
CREATE DATABASE turkFR;

Create a local user:
GRANT ALL ON turkFR.* TO khealey@'localhost' IDENTIFIED BY 'Bib96?reply';

Create a remote access user so you can access the database from your local computer to analyse the data:
GRANT ALL ON turkFR.* TO khealey@'35.8.48.32' IDENTIFIED BY 'Bib96?reply';

Some notes to aid in writing methods and ensure reproducabilirty:

Algorythm for scoring typed realls:

First participants responses are converted to lower case and stripped of any white space. Next, the response is compared
to all the words that the participant has seen so far in the experiment (which are also lowercase and free of white space);
if it matches, it is scored as a correct recall or a prior list intrusion depending on when the word was presented. If the
response does not match a presented word, we check if it exactly matches any of the 235886 words in Webster's Second International
dictionary plus suplemental (https://libraries.io/npm/web2a). If so, it is scored as a ELI. If it does not match any word
in the dictionary, it is assumed to be a typo and we attempt to correct its spelling by computing the Damerau–Levenshtein
distance \cite{Dame64} between the response and each word in the dictionary. Because almost all responses corresponde to
words that were presented (i.e., ELIs are rare), we do not assume the response corresponds to the closest word in the
dictionary. Instead we find the distance between the response and the closest word that was presented  and compare that
distance to the distribution of distances from all words in the dictionary. If it is below the tenth percentile of that
distribution (i.e., closer than 90% of the words in the dictionary) it is assumed to be that list item, otherwise it is
assumed to be an ELI.


# needed for edit distance:
 pip install pyxDamerauLevenshtein




