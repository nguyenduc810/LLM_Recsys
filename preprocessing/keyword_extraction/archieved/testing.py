

import pandas as pd 
from src.utils import dataloader


loader = dataloader.ReviewLoader()
singapore = loader.load_data(city='singapore')

rvs = list(singapore['text'])


# try extracting food name 

# with spacy 
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The pizza was delicious, but the pasta was too salty."

doc = nlp(text)
doc.ents
for ent in doc.ents:
    print(ent.text)
    if ent.label_ == "FOOD":
        print("---")
        print(ent.text)
        print("---")


# with nltk. seems like can use rule base to extract? 
import nltk

text = "The pizza was delicious, but the butternut squash pasta was too salty."

tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)
chunked = nltk.ne_chunk(tagged)
tokens
tagged
chunked
for tree in chunked:
    print(tree)
    print("----")
    if hasattr(tree, "label") and tree.label() == "FOOD":
        print(" ".join([leaf[0] for leaf in tree.leaves()]))


# textblob 
from textblob import TextBlob

text = "The pizza was delicious, but the pasta was too salty."

blob = TextBlob(text)
blob
for np in blob.noun_phrases:
    print(np)
    if "food" in np:
        print("---")
        print(np)

blob = TextBlob(rvs[0].lower())
a = rvs[0].lower()
blob.noun_phrases.count('buger fix')
blob.word_counts('buger fix']

a
for np in blob.noun_phrases:
    print (np)

for words, tag in blob.tags:
    print (words, tag)

rvs[0]

wiki = TextBlob("python is a high-level, general-purpose programming language.")
wiki.noun_phrases
wiki.tags

users = singapore['user_id']  # singapore users 
print(len(set(loader.load_data(city='pittsburgh')['user_id'])))

a = list(set(users))
len(a)


ifile = 'data/input/reviews_all.csv'
dt = pd.read_csv(ifile)

dt.head()

users = list(dt['user_id'])

len(users)
a = list(set(users))
len(a)


