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
finder=FinderAcora()

from collections import defaultdict #pip install collection
import ner #pip install ner
import pyap #pip install pyap
import pandas as pd

from prettytable import PrettyTable #pip install prettytable

import nltk #pip install --user -U nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

import spacy
from spacy.matcher import Matcher
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp_model = en_core_web_sm.load()


# extract text from pdf
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)
 
# extract text from doc
def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None


# append text in each resume_section
def parse_txt_to_section(text):
    text_lower = text.lower()
    # define modules in resume
    module_name = ['experience','project','education','skill']
    # initialize index for each section
    section_ind = {'experience': -1, 'skill': -1, 'education': -1, 'project': -1}
    # initialize text for each section
    section_text = {'personal_info': '', 'experience': '', 'skill':'', 'education': ''}
    
    ind = []
    for module in module_name:
        # append the index for each section title
        if module in list(section_ind.keys()):
            section_ind[module] = text_lower.find(module)
            #print(module, text.find(module))
        ind.append(text_lower.find(module))
    
    if not ind:
        return None
    
    # sort ind
    ind.sort()
    
    # append personal info from input text
    if ind[0] and (ind[0] > 0):
        section_text['personal_info'] = text[:ind[0]] 
    else:
        section_text['personal_info'] = text[:ind[1]] 
    
    # append experience from input text
    if section_ind['experience'] >= 0:
        ind_2 = ind.index(section_ind['experience'])+1 if (ind.index(section_ind['experience'])+1<len(ind)) else -1
        section_text['experience'] = text[section_ind['experience']:(ind[ind_2] if (ind_2 != -1) else ind_2)]
    
        ind_22 = ind.index(section_ind['project'])+1 if (ind.index(section_ind['project'])+1<len(ind)) else -1
        section_text['experience'] += text[section_ind['project']:(ind[ind_22] if (ind_22 != -1) else ind_22)]
    
    
    # append skill from input text
    if section_ind['skill'] >= 0:
        ind_3 = ind.index(section_ind['skill'])+1 if (ind.index(section_ind['skill'])+1<len(ind)) else -1
        section_text['skill'] = text[section_ind['skill']:(ind[ind_3] if (ind_3 != -1) else ind_3)]
    
    # append education from input text
    if section_ind['education'] >= 0:
        ind_4 = ind.index(section_ind['education'])+1 if (ind.index(section_ind['education'])+1<len(ind)) else -1
        section_text['education'] = text[section_ind['education']:(ind[ind_4] if (ind_4 != -1) else ind_4)]
    
    return section_text

def extract_names(text):
    """person_names = []
 
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                person_names.append(
                    ' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
                )
                
    if not person_names:"""
    temp = text.split()
    person_names = [temp[0] + ' ' + temp[1]]
    
    return person_names[0]
    
def find_email(text):  
    results = re.findall('\S+@\S+', text)  
    emails = ""
    for x in results:
        emails += str(x)
    return emails

def find_phone(text): 
    try:
        results = re.findall('((?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}))', text)  
        x=""
        for s in results[0]:
            if s.isdigit():
                x+=s
        phone = int(x)
        return phone
    except:
        return ""

def extract_skills(text, skills_db):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(text)
 
    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]
 
    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]
 
    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
 
    # we create a set to keep the results in.
    found_skills = set()
 
    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in skills_db:
            found_skills.add(token)
 
    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.lower() in skills_db:
            found_skills.add(ngram)
 
    return found_skills
    

def extract_education(text):

    YEAR = r'(((20|19)(\d{2})))'
    EDUCATION = ['be', 'b.e.', 'b.e', 'bs', 'b.s', 'me', 'm.e', 'm.e.', 'ms', 'm.s', 'btech', 'mtech', \
            'ssc', 'hsc', 'cbse', 'icse', 'x', 'xii', 'phd', 'master', 'bachelor']
    
    DICT_SCHOOL = [
    'school',
    'college',
    'university',
    'academy',
    'faculty',
    'institute',
    'faculdades',
    'Schola',
    'schule',
    'lise',
    'lyceum',
    'lycee',
    'polytechnic',
    'kolej',
    'ünivers',
    'okul',
    ]

    STOPWORDS = set(stopwords.words('english'))

    out = {'school': [], 'degree': [], 'degree_date':[]}
    # Extract school 
    try:
        for i, tex in enumerate(text.split('\n')):
            tex = re.sub(r'[?|$|.|!|,|(|)]', r'', tex)
            for j, word in enumerate(tex.split(' ')):
                if (word.lower() in DICT_SCHOOL) and (word.lower() not in STOPWORDS) and (tex not in out['school']):
                    out['school'].append(" ".join(tex.split()))
    except IndexError:
        pass

    # Extract school education degree
    try:
        for i, tex in enumerate(text.split('\n')):
            tex = re.sub(r'[?|$|.|!|,|(|)]', r'', tex)
            for j, word in enumerate(tex.split(' ')):
                if word.lower() in EDUCATION and word.lower() not in STOPWORDS:
                    out['degree'].append(word)
    except IndexError:
        pass
    
    # find dates
    matches_date = datefinder.find_dates(text)
    
    
    dt2 = list(matches_date)
    for date in dt2:
        #print(date.strftime('%Y-%m-%d'))
        out['degree_date'].append(date.strftime('%Y-%m-%d'))
    
    return out    


def extract_experiences(text):
    
    text = text.replace("-", "aaa" )
    out = defaultdict(list)
    
    # match for job title
    try:
        matches = finder.findall(text)



        # process all date data to be used in 'append dates' section
        # find all dates
        matches_date = list(datefinder.find_dates(text, source=True, index=True))

        date_ind=[]
        for i, date in enumerate(matches_date):
            numm = [s for s in date[1].split() if s.isdigit()]
            # append to date_ind with conditions
            if numm and (len(numm[0]) >= 4) and (int(numm[0]) >= 1900):
                date_ind.append((date[2][0],i))


        # append to dict
        for i, match in enumerate(matches):
            # find front and end index of text
            ind_front = max(int(match[0])-100,0)
            ind_end = min(int(match[0])+100,len(text))

            # append job title
            out['experience'+str(i)].append(match[2])

            # append company name
            ner_list = nlp_model(text[ind_front:ind_end])
            potential_comp_name = [X.text for X in ner_list.ents if X.label_=='ORG']
            out['experience'+str(i)].append(potential_comp_name)

            # append dates
            try:
                # find the nearest two dates with job title index
                ind_near = nearest_n_ind(match[0], date_ind, 2)

                # append dates to experience dict
                dt1 = matches_date[ind_near[0]][0]
                out['experience'+str(i)].append(dt1.strftime('%Y-%m-%d'))

                # append second date if the second date is greater than the first date
                if matches_date[ind_near[1]][0] >= matches_date[ind_near[0]][0]:
                    dt2 = matches_date[ind_near[1]][0]
                    out['experience'+str(i)].append(dt2.strftime('%Y-%m-%d'))

                else:
                    dt2 = datetime.today()
                    out['experience'+str(i)].append(dt2.strftime('%Y-%m-%d'))

                # append experience duration
                date_length = (dt2-dt1)
                year_exp = date_length.days/365
                out['experience'+str(i)].append(year_exp)

            except:
                pass
    except:
        return out

    return out

def experience_year(list1):
    exp_set = []
    # find unique job experience title
    for key in list1.keys():
        exp_set.append(list1[key][0])
    
    # append experience's duration
    experience_year = defaultdict(list)
    try:
        for key in list1.keys():
            if not experience_year[str(list1[key][0])]:
                experience_year['duration: ' + str(list1[key][0])] = 0
            if list1[key][0] in set(exp_set):
                experience_year['duration: ' + str(list1[key][0])] += (list1[key][4])
    except:
        return experience_year
    
    return experience_year
    
def nearest_n_ind(ind_target, ind_list, n):
    minus = abs(np.array(ind_list) - ind_target)
    minus_set = [(mi[0],li) for mi, li in zip(minus,ind_list)]
    minus_set.sort()
    return_set = [ele[1][1] for ele in minus_set]
    return return_set[:n]

def parse_module(section_text):
    # parse personal info
    personal_info = {'name': '', 'phone': '', 'email': '', 'address':''}
    personal_info['name'] = extract_names(section_text['personal_info'])
    personal_info['email'] = find_email(section_text['personal_info'])
    personal_info['phone'] = find_phone(section_text['personal_info'])
    personal_info['address'] = re.findall("[0-9]{1,3} .+ .+ [A-Za-z]{2} [0-9]{5}", section_text['personal_info'])
    
    
    # parse education
    education = extract_education(section_text['education'])
    
    # parse experience
    experience = extract_experiences(section_text['experience'])
    if experience:
        experience_duration = experience_year(experience)
    else:
        experience_duration = defaultdict(list)
    
    # parse skills
    data = pd.read_csv('/Users/fangningzheng/Insync/fangninz@usc.edu/GDrive/career/HireBeat/week_2_ResumeParser/resumeParser/skill_12_12_2022_Master_v5.csv')
    skills_db = list(data.iloc[:, 1])
    skill = extract_skills(section_text['experience'], skills_db)
    skill.union(extract_skills(section_text['skill'], skills_db))
    skill_dict = {'skill':list(skill)}
    
    # merge dictionaries
    resume_section = {**personal_info, **education, **experience, **experience_duration, **skill_dict}
    
    return resume_section


