import spacy 
import pandas as pd 
import sys 
sys.path.append('/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/recsys_coldstart')
from src.utils import dataloader
from textblob import TextBlob
from pprint import pprint
from tqdm import tqdm 
import spacy
import os 
import json


def update_np_count(blob, p2c):
    for p in blob.noun_phrases:
        p2c[p] = p2c.get(p, 0) + blob.noun_phrases.count(p)


def get_noun_phrases(doc, output=None, keep=None):
    if keep is None:
        return list([np.text.lower() for np in doc.noun_chunks])
    if output is None:
        output = {}
    for nc in doc.noun_chunks:
        ws = []
        for word in nc:
            if word.pos_ in keep:
                ws.append(word.text.lower())
        if len(ws) > 0:
            n = ' '.join(ws)
            output[n] = output.get(n, 0) + 1
    return output


def print_details(doc):
    for token in doc:
        print(f'{token.text:{8}} {token.pos_:{6}} {token.tag_:{6}} {token.dep_:{6}} {spacy.explain(token.pos_):{20}} {spacy.explain(token.tag_)}')


def update_dict(global_dict, new_dict):
    for k, v in new_dict.items():
        global_dict[k] = global_dict.get(k, 0) + v


def increase_count_with_list(global_dict, new_list):
    for k in new_list:
        global_dict[k] = global_dict.get(k, 0) + 1


def extract_raw_keywords_for(city, loader, odir, np2count, np2review_count, np2res_count):
    print("Processing for", city)
    ofile = os.path.join(odir, '{}.json'.format(city))
    dt_city = loader.load_data(city=city)
    rids = list(set(list(dt_city['rest_id'])))
    counter = 0
    for rid in tqdm(rids):  # for each restaurant id 
        counter += 1
        if counter <= 900: 
            continue 
        r1s = loader.load_data(data=dt_city, rest_id=rid)
        rest_nps = [] 
        # for r in tqdm(list(r1s['text'])):  # for each review 
        for r in list(r1s['text']):  # for each review 
            doc = nlp(r)
            tmp = get_noun_phrases(doc, keep=keep)  # np for this review 
            nps = list(set(tmp.keys()))
            increase_count_with_list(np2review_count, nps)
            update_dict(np2count, tmp)
            rest_nps += nps
        increase_count_with_list(np2res_count, list(set(rest_nps)))
    output = {"np2count": np2count, "np2review_count": np2review_count, 'np2res_count': np2res_count}
    json.dump(output, open(ofile, 'w'))
    print("Saved to", ofile)

nlp = spacy.load("en_core_web_sm")
keep=['ADJ', 'NOUN', 'PROPN', 'VERB']
odir = 'data/ke_spacy'
cities = ['phoenix', 'london', 'lasvegas', 'edinburgh', 'charlotte', 'pittsburgh', 'singapore']

loader = dataloader.ReviewLoader(data_file='data/input/reviews_all.csv', split_file='data/input/users_split_1.json')

dt = json.load(open('/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/recsys_coldstart/data/ke_spacy/phoenix.json'))
np2count = dt['np2count']
np2rev = dt['np2review_count']
np2res = dt['np2res_count']

extract_raw_keywords_for('phoenix', loader, odir, np2count, np2rev, np2res)

    

# city_name = 'lasvegas'
# list(set(list(loader.data['city'])))
# ofile = os.path.join(odir, '{}.json'.format(city_name))

# dt_city = loader.load_data(city=city_name)
# rids = list(set(list(dt_city['rest_id'])))
# np2count = {}   # frequency 
# np2review_count = {}  # #reviews 
# np2res_count = {}  # #restaurants containing this noun phrase 
# counter = 0
# for rid in tqdm(rids):  # for each restaurant id 
#     r1s = loader.load_data(data=dt_city, rest_id=rid)
#     rest_nps = [] 
#     for r in tqdm(list(r1s['text'])):  # for each review 
#         doc = nlp(r)
#         tmp = get_noun_phrases(doc, keep=keep)  # np for this review 
#         nps = list(set(tmp.keys()))
#         increase_count_with_list(np2review_count, nps)
#         update_dict(np2count, tmp)
#         rest_nps += nps
#     increase_count_with_list(np2res_count, list(set(rest_nps)))
#     counter += 1
#     if counter > 3:
#         break

# output = {"np2count": np2count, "np2review_count": np2review_count, 'np2res_count': np2res_count}
# json.dump(output, open(ofile, 'w'))
# print("Saved to", ofile)