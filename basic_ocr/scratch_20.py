import os
from datetime import datetime
from difflib import SequenceMatcher
from hashlib import md5
import string
start = datetime.now()
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

stopwords = stopwords.words('english')

#sample_dir = '/home/rhallman/test/'
sample_dir = '/home/rhallman/test/SampleRequests-Younce'
#sample_files = ['flores.txt', 'gonzalez.txt', 'smith.txt']
sample_files = [ x for x in os.listdir(sample_dir) if x.startswith('test2') ]

def is_date_obj(text_string):
    supported_formats = [
        '%m/%d/%Y', '%m/%d/%y', '%m-%d-%y', '%m-%d-%Y',
        '%m%d%Y', '%m%d%y', '%Y-%m-%d', '%Y%m%d', '%b %m %Y',
        '%B %m %Y', '%b %m, %Y', '%B %m, %Y'
    ]
    r = None
    for sformat in supported_formats:
        try:
            r = datetime.strptime(text_string, sformat)
        except:
            continue
    if r:
        return True
    else:
        return False

def hash_file(file_path):
    output = []
    modified_text = []
    with open(file_path, 'r') as f:
        modified_lines = []
        lines = f.readlines()
        for line in lines:
            for res in line.split('    '):
                modified_lines.append(res)

        for line in modified_lines:
            line = line.strip()
            if is_date_obj(line):
                continue
            modified_line = []
            for word in line.split(" "):
                word = "".join([letter for letter in word if letter not in string.punctuation]).strip()
                if word in stopwords:
                    continue
                if is_date_obj(word):
                    continue
                if not word:
                    continue
                modified_line.append(word)
            if modified_line and modified_line != '\n':
                modified_line = " ".join(modified_line).strip()
                output.append(md5(modified_line.encode('utf-8')).hexdigest())
                modified_text.append(modified_line)
    return output, modified_text

match_success_list = []
results = {}
full_text_ratios = []
for sample_file in sample_files:
    file_path = os.path.join(sample_dir,sample_file)
    results[sample_file] = hash_file(file_path)

for source_file in sample_files:
    print("Results for %s (%s lines):" % (source_file, len(results[source_file][0])))
    if source_file in ['test2_new_78c7aa40-e911-4553-8074-e02afab42da7_69f9530132fa412fba5e4bbca53f7dd8-1.txt','test2_new_db1b5112-fa35-462f-bc52-146d331d82e7_e2738eaddc79460a8e64527e60316374-1.txt']:
        with open(source_file, 'w') as f:
            f.write("\n".join(results[source_file][1]))
    for comparison_file in sample_files:
        if comparison_file == source_file:
            continue
        matches = len([x for x in results[source_file][0] if x in results[comparison_file][0]])
        print('    %s: %s' % (comparison_file, matches))
        match_success_list.append(matches / len(results[source_file][0]))

        full_text_ratios.append(SequenceMatcher(None, " ".join(results[source_file][1]), " ".join(results[comparison_file][1])).ratio())
print('\nSuccess rate: %s' % (sum(match_success_list) / len(match_success_list)))
print('Full Text Results: %s' % (sum(full_text_ratios) / len(full_text_ratios)))
res = set(results[list(results.keys())[0]][0])
for key, item in results.items():
    res.intersection_update(item[0])
print('Found %s common lines' % len(res))
stop = datetime.now()

print(stop-start)