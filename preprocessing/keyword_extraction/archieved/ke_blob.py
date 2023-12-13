'''

'''

import pandas as pd 
from src.utils import dataloader
from textblob import TextBlob
from pprint import pprint
from tqdm import tqdm 
import spacy

def update_np_count(blob, p2c):
    for p in blob.noun_phrases:
        p2c[p] = p2c.get(p, 0) + blob.noun_phrases.count(p)


loader = dataloader.ReviewLoader()
singapore = loader.load_data(city='lasvegas')
rids = list(set(list(singapore['rest_id'])))
r1s = loader.load_data(data=singapore, rest_id=rids[0])

# ---
r = list(r1s['text'])[0]
blob = TextBlob(r)
for p in blob.noun_phrases:
    print("{}:\t{}".format(p, blob.noun_phrases.count(p)))


nlp = spacy.load("en_core_web_sm")
doc = nlp(r)
for token in doc:
    print(f'{token.text:{8}} {token.pos_:{6}} {token.tag_:{6}} {token.dep_:{6}} {spacy.explain(token.pos_):{20}} {spacy.explain(token.tag_)}')
for d in doc.ents:
    print(d)
doc.
blob.pos_tags
r
p2c = {}

for r in tqdm(list(r1s['text'])):
    blob = TextBlob(r)
    for p in blob.noun_phrases:
        p2c[p] = p2c.get(p, 0) + blob.noun_phrases.count(p)

# --- 
r1s = loader.load_data(data=singapore, rest_id=rids[0])
len(r1s)
p2c = {}

for r in tqdm(list(r1s['text'])):
    blob = TextBlob(r)
    for p in blob.noun_phrases:
        p2c[p] = p2c.get(p, 0) + blob.noun_phrases.count(p)

# ---
p2c = {}
for rid in tqdm(rids): 
    r1s = loader.load_data(data=singapore, rest_id=rid)
    # print("#reviews: {}".format(len(r1s)))
    blob = TextBlob(' '.join(list(r1s['text'])))
    for p in blob.noun_phrases:
        p2c[p] = p2c.get(p, 0) + blob.noun_phrases.count(p)


p2c_sorted = sorted(p2c.items(), key=lambda x: x[1], reverse=True)
pprint(p2c_sorted[:200])

# --- 



rvs = list(singapore['text'])

alldoc = ' '.join(rvs[:1000])

blob = TextBlob(alldoc)
p2c = {}
for p in blob.noun_phrases:
    p2c[p] = blob.noun_phrases.count(p)
p2c_sorted = sorted(p2c.items(), key=lambda x: x[1], reverse=True)
p2c_sorted[:10]

len(blob.noun_phrases)
type(blob.noun_phrases)

blob.tags