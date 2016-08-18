# name of the file you want to save the pool to
save_file = '/Users/khealey/psiturk_exps/turkFR/static/js/nonegative_word_pool.js'

# load the word pool file and select only those that are not on the list of excluded words (high negative valence)
words = open("wasnorm_wordpool.txt", "r")
excluded = open("excluded_words.txt", "r")
lines = words.read().split()
excluded_lines = excluded.read().split()
good_words = [x for x in lines if x not in excluded_lines]

# format for java
formatted_words = ['["' + item + '"],' for item in good_words]
formatted_words = ['var stims = ['] + formatted_words + ['];']

# print to a file
thefile = open(save_file,'w')
for item in formatted_words:
    thefile.write("%s\n" % item)
thefile.close()

