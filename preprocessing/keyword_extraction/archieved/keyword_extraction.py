'''
Extracting keywords from review content. Use RAKE NLTK since it is more customizable 

- compute stopwords for food: idf? 

https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f
https://csurfer.github.io/rake-nltk/_build/html/index.html
'''

from rake_nltk import Rake 
import nltk
import json
import os 
from tqdm import tqdm 
from nltk import word_tokenize
import pandas as pd 
import numpy as np
from pprint import pprint
from gensim.summarization import keywords 


def load_reviews_as_list(ifile):
    dt = json.load(open(ifile))
    rvs = [] 
    for k, v in dt.items():
        rvs.append(v['text'])
    return rvs


def load_stopwords(ifile, start_idx=1):
    with open(ifile, 'r') as reader:
        tmp = [a.strip() for a in reader.readlines()[start_idx:]]
    return tmp 


def word_only_tokenizer(txt):
    tmp = nltk.tokenize.wordpunct_tokenize(txt)
    return [a for a in tmp if a.isalpha()]

def ext_keywords_rake_nltk(rv, raker_nltk):
    raker_nltk.extract_keywords_from_text(rv)
    return raker_nltk.get_ranked_phrases_with_scores()

def extract_and_count_keywords_for_text(txt, raker_nltk, tk2count=None, tk2doc_count=None, rake_threshold=2):
    if tk2count is None:
        tk2count = {}
    tk_with_scores = ext_keywords_rake_nltk(txt, raker_nltk=raker_nltk)
    # print(tk_with_scores)
    kws = []
    for sc, tk in tk_with_scores:
        if sc >= rake_threshold:
            tk2count[tk] = tk2count.get(tk, 0) + 1
            kws.append(tk)
            # print(tk)
    # update idf 
    if tk2doc_count is not None: 
        for kw in set(kws):
            tk2doc_count[kw] = tk2doc_count.get(kw, 0) + 1
    return tk2count, tk2doc_count

def init_raker_nltk(stop_file='data/stopwords.txt', word_tokenizer=word_only_tokenizer, start_idx=1):
    stopwords = load_stopwords(stop_file, start_idx=start_idx)
    return Rake(stopwords=stopwords, word_tokenizer=word_tokenizer) 

def sort_dict_by_value(idict, reverse=True):
    return sorted(idict.items(), key=lambda x: x[1], reverse=reverse)

def get_reviews_for_city_setname(city, setname, data, splits, key='user_id'):
    return get_reviews_having_ids(data, input_list=splits[city][setname], key=key)

def get_reviews_having_ids(data, input_list, key='user_id'):
    return data[data[key].isin(input_list)]

def extract_and_count_keywords_for_list_of_reviews(reviews, raker_nltk, tk2count=None, tk2doc_count=None, rake_threshold=2):
    if tk2count is None: 
        tk2count = {}  # {token: count}
    if tk2doc_count is None: 
        tk2doc_count = {}  # {token: #reviews containing the keywords}
    for rev in tqdm(reviews): 
        tk2count, tk2doc_count = extract_and_count_keywords_for_text(rev, raker_nltk, tk2count, tk2doc_count, rake_threshold=rake_threshold)
    return tk2count, tk2doc_count, len(reviews)


def get_idf(doc_count, total_doc):
    return np.log(total_doc/(1+doc_count))

def compute_idf(tk2doc_count, total_doc):
    tk2idf_v = {}
    for tk, doccount in tk2doc_count.items():
        tk2idf_v[tk] = get_idf(doccount, total_doc)
    return tk2idf_v


def compute_tf(tk2count):
    total = np.sum(list(tk2count.values()))
    tk2idf = {}
    for tk, c in tk2count.items():
        tk2idf[tk] = c/total
    return tk2idf 

def compute_tfidf(tk2tf, tk2idf):
    tk2s = {}
    for tk, tf in tk2tf.items():
        tk2s[tk] = tf * tk2idf[tk]
    return tk2s


def get_ids_from_all_cities_for_set(splits, setname):
    ids = []
    for k, v in splits.items():
        ids += v[setname]
    return ids


def run_computing_tfidf():
    pass 


# test with splits for separate city, each restaurant is a doc (combining all the reviews)
setname = 'train'
split_file = 'data/input/users_split_1.json'
dt_file = 'data/input/reviews_all.csv'

splits = json.load(open(split_file))
data = pd.read_csv(dt_file)

ids_train_sing = splits['singapore']['train']

data_train_sing = data[data['user_id'].isin(ids_train_sing)]
train_sing_rids = list(set(list(data_train_sing['rest_id'])))  # all restaurants id 
docall = ' '.join(data_train_sing['text'])
tmp = list(data_train_sing['text'])[0]
a = list(data_train_sing[data_train_sing['rest_id'].isin([train_sing_rids[3]])]['text'])
atxt = '\n\n'.join(a)
print(atxt)

stopwords=load_stopwords('/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/recsys_coldstart/data/input/stopwords.txt')
# stopwords += ['card']
rake = Rake(stopwords=stopwords, max_length=5)

rake.extract_keywords_from_text(atxt)
rake.frequency_dist
a = rake.get_ranked_phrases_with_scores()
pprint(a[:100])
rake.degree
rake.

rake.frequency_dist
rake.degree
rake.extract_keywords_from_text(tmp)
print(rake.get_ranked_phrases_with_scores())
rake._tokenize_sentence_to_words(tmp)
rake.extract_keywords_from_text(tmp)

rake.max_length
rake.frequency_dist
rake.min_length
rake.stopwords
a = rake.get_ranked_phrases_with_scores()
a
ws = word_only_tokenizer(tmp)
ws
ws2 = word_tokenize(tmp)
ws2
tmp

# keywords compute for each restaurant 
raker_nltk = init_raker_nltk(stop_file='data/input/stopwords.txt', word_tokenizer=word_only_tokenizer)
kws2 = ext_keywords_rake_nltk(docall, raker_nltk)
json.dump(kws2, open('data/rake-res_as_doc-2-stopwords-our_tokenizer.json', 'w'))
pprint(kws2[:100])

tmpr = data_train_sing[data['rest_id'].isin([train_sing_rids[1]])]['text']
len(tmpr)

rdoc = ' '.join(tmpr)

kws = ext_keywords_rake_nltk(rdoc, raker_nltk)
pprint(kws)

r2 = Rake()
kws2 = ext_keywords_rake_nltk(docall, r2)
pprint(kws2)
json.dump(kws2, open('data/rake-res_as_doc-1.json', 'w'))



len(train_sing_rids)

len(data_train_sing)

data.keys()


raker_nltk = init_raker_nltk(stop_file='data/input/stopwords.txt', word_tokenizer=word_only_tokenizer)

# get all train ids 
train_ids = get_ids_from_all_cities_for_set(splits, 'train')
dt_train = get_reviews_having_ids(data, input_list=train_ids, key='user_id')

# compute tfidf 
tk2count, tk2doc_count, count = extract_and_count_keywords_for_list_of_reviews(dt_train['text'], raker_nltk, rake_threshold=2)
tk2tf = compute_tf(tk2count)  # compute tf 
tk2idf = compute_idf(tk2doc_count, count)  # compute idf 
tk2tfidf = compute_tfidf(tk2tf, tk2idf)  # compute tfidf 
sorted_tfidf = sorted(tk2tfidf.items(), key=lambda x: x[1], reverse=True)  # sort tfidf 
pprint(sorted_tfidf[1000:2000])
len(tk2count)

dt = {"tk2count": tk2count, "tk2doc_count": tk2doc_count, "#reviews": count}
json.dump(dt, open("data/input/reviews_all_token_count.json", 'w'))

sorted_count = sorted(tk2count.items(), key=lambda x: x[1], reverse=True)
pprint(sorted_count[2000:3000])
pprint(sorted_count[:100])


# # ----------------------
# # draft below 
# city = 'singapore'
# uids = splits[city][setname]  # train user ids 
# dt_train = get_reviews_for_city_setname(city, setname, data, splits)
# tk2count, tk2doc_count, count = extract_and_count_keywords_for_list_of_reviews(dt_train['text'], raker_nltk, rake_threshold=2)


# # compute idf 
# tk2idf = compute_idf(tk2doc_count, count)

# # compute tf
# tk2tf = compute_tf(tk2count)


# sorted_tfidf = sorted(tk2tfidf.items(), key=lambda x: x[1], reverse=True)
# pprint(sorted_tfidf[:100])
# pprint(sorted_tfidf[-100:])

# tk2idf['chicken']


# sorted_idf = sorted(tk2idf.items(), key=lambda x:x[1], reverse=False)
# pprint(sorted_tfidf[-100:])


# len(dt_train2)
# dt_train = data[data['user_id'].isin(uids)]
# train_reviews = list(dt_train['text'])
# train_reviews[:2]
# len(data[data['user_id'].isin(splits[city]['test'])])

# for city in splits.keys():
#     uids = splits[city][setname]  # train user ids 
#     dt_train = data[data['user_id'].isin(uids)]


# data.keys()

# splits.keys()
# train = splits['singapore']['train']
# train[:10]

# raker_nltk = init_raker_nltk(stop_file='data/input/stopwords.txt', word_tokenizer=word_only_tokenizer)
# idir = '/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users/singapore/extracted_reviews'
# w2c = extract_and_count_keywords_for_dir(idir=idir, raker_nltk=raker_nltk, rake_threshold=2)
# sw2c = sort_dict_by_value(w2c)
# sw2c[:100]


# # test for a dir 
# raker_nltk = init_raker_nltk(stop_file='data/input/stopwords.txt', word_tokenizer=word_only_tokenizer)
# idir = '/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users/singapore/extracted_reviews'
# w2c = extract_and_count_keywords_for_dir(idir=idir, raker_nltk=raker_nltk, rake_threshold=2)
# sw2c = sort_dict_by_value(w2c)
# sw2c[:100]


# # test for single text 
# rvs = load_reviews_as_list('/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users/lasvegas/extracted_reviews/_0x7W6fizaPP76xNBxBLAQ.json')
# len(rvs)

# txt = rvs[9]
# sents = sent_tokenize(txt)

# raker_nltk.extract_keywords_from_sentences(sents)
# raker_nltk.get_ranked_phrases_with_scores()

# raker_nltk.extract_keywords_from_text(txt)
# raker_nltk.get_ranked_phrases_with_scores()

# tk_with_scores = ext_keywords_rake_nltk(rvs[9], raker_nltk=raker_nltk)

# rvs[9]
# tk_with_scores







# w2c = extract_and_count_keywords_for_file(rv_file, raker_nltk=raker_nltk)
# sw2c[500:1000]
# len(w2c)
# rvs = load_reviews_as_list(rv_file)
# w2c = {} 
# for rv in rvs: 
#     w2c = extract_and_count_keywords_for_text(rv, raker_nltk, w2c)

# sorted_w2c = sorted(w2c.items(), key=lambda x:x[1], reverse=True)
# sorted_w2c[:100]

# words = ext_keywords_rake_nltk(rvs[10])

# w2c = extract_and_count_keywords_for_text(rvs[10], raker_nltk=raker_nltk)
# w2c

# -----
# below is draft 
# import RAKE 
# import src.preprocessing.keyword_extraction.prepare_finegrained_keywords as ke_spacy
# def print_extractions(idx):
#     print_extractions_for_txt(rvs[idx])

# def print_extractions_for_txt(rv):
#     print(rv)
#     print("RAKE NLTK")
#     print(ext_keywords_rake_nltk(rv))
#     print("\nRAKE")
#     print(ext_keywords_rake(rv))
#     print("\nSPACY")
#     print(ke_spacy.extract_noun_phrases(rv, lemma=True))
# def ext_keywords_rake(rv):
#     return raker.run(rv)
# rv_file = '/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users/lasvegas/extracted_reviews/_0x7W6fizaPP76xNBxBLAQ.json'
# stop_path = 'data/stopwords.txt'
# stopwords = load_stopwords(stop_path)
# stopwords
# raker = RAKE.Rake(stop_path)
# r = Rake(stopwords=stopwords, word_tokenizer=word_only_tokenizer)

# txt = "Ordered the filet kabob plate. I eat burger. I couldn't give 5 stars because I ordered my filet medium rare, and there was zero pink inside. Needless to say they gave me well done meat. Not gonna bother sending it back, after it's a fast food type joint.  But I will come back and try it again and hopefully they get it right next time. The hummus is amazing, Definitely a must try place."
# r.extract_keywords_from_text(txt)
# r.get_ranked_phrases_with_scores()
# stopwords
# rvs = load_reviews_as_list(rv_file)
# print_extractions(189)

# print_extractions_for_txt("Ordered the filet's kabob plate. I eat burger. I couldn't give 5 stars because I ordered my filet medium rare, and there was zero pink inside. Needless to say they gave me well done meat. Not gonna bother sending it back, after it's a fast food type joint.  But I will come back and try it again and hopefully they get it right next time. The hummus is amazing, Definitely a must try place.")

# rvs[16]
# r
# for r in rvs:
#     print(r)
#     break
#     if r.startswith('O'):
#         print(r)
#     break


# import re
# pt = "[\W\n] +"
# rev = rvs[189]
# rev
# a = re.split(pt, rev)
# a

# a = nltk.tokenize.word_tokenize(rev)
# words=[word.lower() for word in a if word.isalpha()]
# words
# sents = nltk.tokenize.sent_tokenize(rev)
# sents
# a
