import numpy as np
import re
import os
import json

from pdfminer.high_level import extract_text # pip install pdfminer.six 
import docx2txt

import subprocess #pip install subprocess.run
import datefinder #pip install datefinder
from datetime import datetime
from find_job_titles import FinderAcora #pip install find-job-titles

from collections import defaultdict #pip install collection
import ner #pip install ner
import pyap #pip install pyap
import pandas as pd

from prettytable import PrettyTable #pip install prettytable

import nltk #pip install --user -U nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.corpus import stopwords


import spacy
from spacy.matcher import Matcher
from spacy import displacy
from collections import Counter
import en_core_web_sm

from Utilities import extract_text_from_pdf, extract_text_from_docx, parse_txt_to_section, extract_names, find_email, find_phone, extract_skills, extract_education, extract_experiences, experience_year, nearest_n_ind, parse_module


# initial the big dictionary
resume_info_all = defaultdict(list)

# get filenames
path_main_dir = ""
path_to_doc = path_main_dir + "/resume_examples"
os.chdir(path_to_doc)
files = [f for f in os.listdir('.') if os.path.isfile(f) and (f.endswith('.docx') or f.endswith('.pdf'))]


for i, file in enumerate(files):
    # convert doc/pdf to txt
    os.chdir(path_to_doc)
    if (file[0] not in '~') and file.endswith('.docx'):
        text = extract_text_from_docx(file)

    elif (file[0] not in '~') and file.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    
    # get section text
    section_text = parse_txt_to_section(text)
    
    # get section info into dict
    resume_section = parse_module(section_text)

    
    for key, value in resume_section.items():
        # append value to the correct index
        if key not in resume_info_all.keys():
            resume_info_all[key] = [None] * i
            resume_info_all[key].insert(i, value)
            
        else:
            resume_info_all[key].insert(i, value)
            
    for key in resume_info_all.keys():
        if not key in resume_section.keys():
            resume_info_all[key].insert(i, None)


# save dict to csv file
df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in resume_info_all.items() ]))
pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in resume_info_all.items() ]))
df.to_csv(path_main_dir + 'out_RP.csv')  


# save to json file
# os.chdir(path_main_dir)
#with open("resumeparser_out.json", "w") as outfile:
#    json.dump(resume_info_all, outfile, indent=4)

