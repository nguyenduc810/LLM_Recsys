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
from nltk import sent_tokenize
import pandas as pd 
# from gensim.summarization import keywords 


def load_reviews_as_list(ifile):
    dt = json.load(open(ifile))
    rvs = [] 
    for k, v in dt.items():
        rvs.append(v['text'])
    return rvs


def load_stopwords(ifile):
    with open(ifile, 'r') as reader:
        tmp = [a.strip() for a in reader.readlines()[1:]]
    return tmp 


def word_only_tokenizer(txt):
    tmp = nltk.tokenize.wordpunct_tokenize(txt)
    return [a for a in tmp if a.isalpha()]

def ext_keywords_rake_nltk(rv, raker_nltk):
    raker_nltk.extract_keywords_from_text(rv)
    return raker_nltk.get_ranked_phrases_with_scores()

def extract_and_count_keywords_for_text(txt, raker_nltk, tk2count=None, tk2idf=None, rake_threshold=2):
    if tk2count is None:
        tk2count = {}
    tk_with_scores = ext_keywords_rake_nltk(txt, raker_nltk=raker_nltk)
    kws = []
    for sc, tk in tk_with_scores:
        if sc >= rake_threshold:
            tk2count[tk] = tk2count.get(tk, 0) + 1
            kws.append(tk)
    # update idf 
    if tk2idf is not None: 
        for kw in set(kws):
            tk2idf[kw] = tk2idf.get(kw, 0) + 1
    return tk2count, tk2idf


def extract_and_count_keywords_for_file(json_file, raker_nltk, tk2count=None, rake_threshold=2):
    rvs = load_reviews_as_list(json_file)
    if tk2count is None: 
        tk2count = {}
    for rv in rvs: 
        tk2count = extract_and_count_keywords_for_text(rv, raker_nltk, tk2count, rake_threshold=rake_threshold)
    return tk2count

def extract_and_count_keywords_for_dir(idir, raker_nltk, tk2count=None, tk2idf=None, rake_threshold=2):
    if tk2count is None: 
        tk2count = {}  # {token: count}
    if tk2idf is None: 
        tk2idf = {}  # {token: #reviews containing the keywords}
    tmp = os.listdir(idir)
    for fname in tqdm(tmp, total=len(tmp)):
        if fname.startswith('.'):
            continue  # macos issue 
        ifile = os.path.join(idir, fname)
        tk2count = extract_and_count_keywords_for_file(ifile, raker_nltk, tk2count, rake_threshold=rake_threshold)
    return tk2count


def init_raker_nltk(stop_file='data/stopwords.txt', word_tokenizer=word_only_tokenizer):
    stopwords = load_stopwords(stop_file)
    return Rake(stopwords=stopwords, word_tokenizer=word_tokenizer) 


def sort_dict_by_value(idict, reverse=True):
    return sorted(idict.items(), key=lambda x: x[1], reverse=reverse)


def get_reviews_for_city_setname(city, setname, data, splits, key='user_id'):
    return data[data[key].isin(splits[city][setname])]


def extract_and_count_keywords_for_list_of_reviews(reviews, raker_nltk, tk2count=None, tk2idf=None, rake_threshold=2):
    if tk2count is None: 
        tk2count = {}  # {token: count}
    if tk2idf is None: 
        tk2idf = {}  # {token: #reviews containing the keywords}
    for rev in tqdm(reviews): 
        tk2count = extract_and_count_keywords_for_text(rev, raker_nltk, tk2count, tk2idf, rake_threshold=rake_threshold)
    return tk2count, tk2idf, len(reviews)


# test with splits for separate city 
setname = 'train'
split_file = 'data/input/users_split_v2.json'
dt_file = 'data/input/reviews_all.csv'

splits = json.load(open(split_file))
data = pd.read_csv(dt_file)
raker_nltk = init_raker_nltk(stop_file='data/input/stopwords.txt', word_tokenizer=word_only_tokenizer)

city = 'singapore'
uids = splits[city][setname]  # train user ids 
dt_train = get_reviews_for_city_setname(city, setname, data, splits)

tk2count, tk2idf, count = extract_and_count_keywords_for_list_of_reviews(dt_train, raker_nltk, rake_threshold=2)


len(dt_train2)
dt_train = data[data['user_id'].isin(uids)]
train_reviews = list(dt_train['text'])
train_reviews[:2]
len(data[data['user_id'].isin(splits[city]['test'])])

for city in splits.keys():
    uids = splits[city][setname]  # train user ids 
    dt_train = data[data['user_id'].isin(uids)]


data.keys()

splits.keys()
train = splits['singapore']['train']
train[:10]

raker_nltk = init_raker_nltk(stop_file='data/input/stopwords.txt', word_tokenizer=word_only_tokenizer)
idir = '/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users/singapore/extracted_reviews'
w2c = extract_and_count_keywords_for_dir(idir=idir, raker_nltk=raker_nltk, rake_threshold=2)
sw2c = sort_dict_by_value(w2c)
sw2c[:100]


# test for a dir 
raker_nltk = init_raker_nltk(stop_file='data/input/stopwords.txt', word_tokenizer=word_only_tokenizer)
idir = '/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users/singapore/extracted_reviews'
w2c = extract_and_count_keywords_for_dir(idir=idir, raker_nltk=raker_nltk, rake_threshold=2)
sw2c = sort_dict_by_value(w2c)
sw2c[:100]


# test for single text 
rvs = load_reviews_as_list('/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users/lasvegas/extracted_reviews/_0x7W6fizaPP76xNBxBLAQ.json')
len(rvs)

txt = rvs[9]
sents = sent_tokenize(txt)

raker_nltk.extract_keywords_from_sentences(sents)
raker_nltk.get_ranked_phrases_with_scores()

raker_nltk.extract_keywords_from_text(txt)
raker_nltk.get_ranked_phrases_with_scores()

tk_with_scores = ext_keywords_rake_nltk(rvs[9], raker_nltk=raker_nltk)

rvs[9]
tk_with_scores







w2c = extract_and_count_keywords_for_file(rv_file, raker_nltk=raker_nltk)
sw2c[500:1000]
len(w2c)
rvs = load_reviews_as_list(rv_file)
w2c = {} 
for rv in rvs: 
    w2c = extract_and_count_keywords_for_text(rv, raker_nltk, w2c)

sorted_w2c = sorted(w2c.items(), key=lambda x:x[1], reverse=True)
sorted_w2c[:100]

words = ext_keywords_rake_nltk(rvs[10])

w2c = extract_and_count_keywords_for_text(rvs[10], raker_nltk=raker_nltk)
w2c

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
