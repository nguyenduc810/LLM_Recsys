'''
Steps
1. Extract the noun phrases using the customized extractor (group consecutive nouns and verbs)
2. Construct a list of keywords (based on frequency)
3. Refine: re-run the extraction with the list from 2

Using spacy seems to be ok, but for some cases, the results are not understandable. E.g.,
S1 = "Pulled pork sandwich, half regular sausage, corn, beans, bread, and banana pudding"
S2 = "pulled pork sandwich, half regular sausage, corn, beans, bread, and banana pudding" (S1.lower())

POS tags of all the words in S1 and S2 are correspondingly the same, but the noun_chunks are very different (correct for S1, but not for S2). Why?
--------------
TextBlob
- Worked for "Best Beef Wellington I've ever eaten. :)" --> beef wellington
- Not fully worked for "Pulled pork sandwich, half regular sausage, corn, beans, bread, and banana pudding" (not extract all)

'''


import pandas as pd
import spacy
from tqdm import tqdm
import os
import re
from collections import OrderedDict
import ast
import json
# import pyperclip
import numpy as np


def get_values(idict, keys):
    values = []
    for k in keys:
        values.append(idict.get(k, ""))
    return values


def get_noun_phrases(doc, keep=None):
    if keep is None:
        return list(set([np.text.lower() for np in doc.noun_chunks]))
    nouns = []
    for nc in doc.noun_chunks:
        ws = []
        for word in nc:
            if word.pos_ in keep:
                ws.append(word.text.lower())
        if len(ws) > 0:
            nouns.append(' '.join(ws))
    return list(set(nouns))


def add_space_after_dot(text):
    # return re.sub(r'(?<=[.,])(?=[^\s])', r' ', text)
    return re.sub(r'(?<=[.,])(?=[a-zA-Z])', r' ', text)


def analyze_file(ifile, odir='data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_chunks_NOUN_PROPN_VERB/',
                 keep=['NOUN', 'PROPN', 'VERB']):
    data = pd.read_csv(ifile)
    ifile_name = os.path.basename(ifile)[:-4]
    noun_ofile = os.path.join(odir, "{}_nouns.csv".format(ifile_name))
    noun_count_ofile = os.path.join(odir, "{}_noun_count.csv".format(ifile_name))

    print("Processing ", ifile)
    print("Saving to ", noun_ofile)
    print("Saving to ", noun_count_ofile)

    captions = data['caption']
    imgs = data['image_id']
    img2nouns = {}
    noun2count = {}

    for img, capt in tqdm(zip(imgs, captions), total=len(imgs)):
        doc = nlp(add_space_after_dot(capt))
        nouns = get_noun_phrases(doc, keep=keep)
        for n in nouns:
            noun2count[n] = noun2count.get(n, 0) + 1
        img2nouns[img] = nouns

    df1 = pd.DataFrame({"image_id": list(imgs), "caption": list(captions), "keywords": get_values(img2nouns, list(imgs))})
    df1.to_csv(noun_ofile)

    df2 = pd.DataFrame({"Keyword": list(noun2count.keys()), "Count": list(noun2count.values())})
    df2.to_csv(noun_count_ofile)


def analyze_file_customized(ifile, odir, lemma=False):
    data = pd.read_csv(ifile)
    ifile_name = os.path.basename(ifile)[:-4]
    noun_ofile = os.path.join(odir, "{}_nouns.csv".format(ifile_name))
    noun_count_ofile = os.path.join(odir, "{}_noun_count.csv".format(ifile_name))

    print("Processing ", ifile)
    print("Saving to ", noun_ofile)
    print("Saving to ", noun_count_ofile)

    captions = data['caption']
    imgs = data['image_id']
    img2nouns = {}
    noun2count = {}

    for img, capt in tqdm(zip(imgs, captions), total=len(imgs)):
        nouns = extract_noun_phrases(capt, lemma=lemma)
        for n in nouns:
            noun2count[n] = noun2count.get(n, 0) + 1
        img2nouns[img] = nouns

    df1 = pd.DataFrame({"image_id": list(imgs), "caption": list(captions), "keywords": get_values(img2nouns, list(imgs))})
    df1.to_csv(noun_ofile)

    df2 = pd.DataFrame({"Keyword": list(noun2count.keys()), "Count": list(noun2count.values())})
    df2.to_csv(noun_count_ofile)

def test(sent):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sent)
    print(list(doc.noun_chunks))
    for w in doc:
        print("{}: {}".format(w, w.pos_))
    print("---")


def is_valid(current_pos, must):
    for pos in current_pos:
        if pos in must:
            return True
    return False


def extract_noun_phrases(sent, verbose=False, lemma=False):
    doc = nlp(sent)
    keep = ['NOUN', 'PROPN', 'VERB']
    must = ['NOUN', 'PROPN']

    nps = []  # noun phrases

    current_words = []
    current_pos = []

    for word in doc:
        if verbose:
            print(word, word.pos_)
        if word.pos_ in keep:
            if lemma:
                current_words.append(word.lemma_.lower())
            else:
                current_words.append(word.text.lower())
            current_pos.append(word.pos_)
        else:
            # save the previous
            if len(current_words) > 0:
                if is_valid(current_pos, must):
                    nps.append(" ".join(current_words))
                current_words = []
                current_pos = []
    if len(current_words) > 0:
        if is_valid(current_pos, must):
            nps.append(" ".join(current_words))
    return list(set(nps))


def get_key2value_df_file(ifile, key, value):
    np_count_df = pd.read_csv(ifile)
    return get_key2value_df(np_count_df, key, value)


def get_key2value_df(np_count_df, key, value, ast_value=False):
    np2count = {}
    for k, v in zip(np_count_df[key], np_count_df[value]):
        if ast_value:
            v = ast.literal_eval(v)
        np2count[k] = v
    return np2count

def tuple2dict(ituple):
    odict = OrderedDict()
    for k, v in ituple:
        odict[k] = v
    return odict


def has_intersection(nouns, keywords):
    for n in nouns:
        if n in keywords:
            return True
    return False

def check_coverage(img2nps, keywords):
    # check how many images covered using the list of keywords
    count = 0
    kws = set(keywords)
    for img, nps in tqdm(img2nps.items()):
        nps = ast.literal_eval(nps)
        if has_intersection(nps, keywords):
            # if len(kws.intersection(nps)) > 0:
            count += 1
    print("(#Keywords: {})\t{}\t[{}/{}]".format(len(keywords), count/len(img2nps), count, len(img2nps)))


def get_top_keywords(all_keywords, min_threshold=1, top=-1, min_len=3, keep_count=False):
    keywords = []
    if keep_count:
        keywords = {}
    for k, v in all_keywords.items():
        if len(keywords) > top > 0 or v < min_threshold:
            break
        if len(k) >= min_len:
            if keep_count:
                keywords[k] = v
            else:
                keywords.append(k)
    return keywords


def refine_keywords(keywords, np2c, threshold=10, debug=None):
    np2new_nps = {}
    for np, count in tqdm(np2c.items()):
        np = str(np)
        if count >= threshold:
            continue
        newkws = []
        if debug and np != debug:
            continue
        for k in keywords:
            p = '^({})\W|\W({})\W|\W({})$'.format(k, k, k)
            res = re.search(p, np)
            if debug:
                print("Pattern: ", p)
                print(res)
            if res is not None:
                newkws.append(k)
        np2new_nps[np] = newkws
    return np2new_nps


def create_pattern(keywords, template=r'^({})\W|\W({})\W|\W({})$|^({})$', separate=False):
    '''
    previous template=r'^({})\W|\W({})\W|\W({})$'
    :param keywords:
    :param template:
    :param separate:
    :return:
    '''
    tmp = []
    for i, k in enumerate(keywords):
        tmp.append(template.format(k, k, k, k))
    if separate:
        return tmp
    return "|".join(tmp)


def get_matched_keywords(capt, pattern):
    tmp = []
    a = re.findall(pattern, capt.lower())
    if a:
        for b in a:
            for c in b:
                if c != '':
                    tmp.append(c)
    return list(set(tmp))

def extract_keywords_using_selected_list(img2capt, img2nps, keywords, redo_all=False):
    '''
    only run for the captions that do not have any keywords matched with the selected ones
    :param img2capt:
    :param keywords:
    :return:
    '''
    pattern = create_pattern(keywords)
    kwset = set(keywords)
    img2kws = {}
    print("Redo all: {}".format(redo_all))
    for img, capt in tqdm(img2capt.items()):
        if redo_all:
            img2kws[img] = get_matched_keywords(capt, pattern)
        else:
            current_nps = ast.literal_eval(img2nps[img])
            if len(kwset.intersection(current_nps)) == 0:
                img2kws[img] = get_matched_keywords(capt, pattern)
            else:
                img2kws[img] = current_nps
    return img2kws


def extract_keywords_using_selected_list_multi_patterns(img2capt, keywords, standardize=False):
    '''
    only run for the captions that do not have any keywords matched with the selected ones
    :param img2capt:
    :param keywords:
    :return:
    '''
    patterns = create_pattern(keywords, separate=True)
    img2kws = {}
    for img, capt in tqdm(img2capt.items()):
        img2kws[img] = extract_keywords_using_selected_list_multi_patterns_for_an_image(capt, patterns, standardize=standardize)
    return img2kws


def standardize_capt(capt):
    ws = nlp(capt)
    ws2 = [w.lemma_.lower() for w in ws]
    return ' '.join(ws2)


def extract_keywords_using_selected_list_multi_patterns_for_an_image(capt, patterns, standardize=False):
    tmp = []
    for pattern in patterns:
        if standardize:
            capt = standardize_capt(capt)
        res = get_matched_keywords(capt, pattern)
        tmp += res
    return list(set(tmp))


def check_empty_keywords(ifile):
    idata = pd.read_csv(ifile)
    count = 0
    for img, kws in zip(idata['image_id'], idata['keywords']):
        kws = ast.literal_eval(kws)
        if len(kws) > 0:
            count += 1
    print("{} [{}/{}]".format(count/len(idata), count, len(idata)))


def run_extracting_keywords_for_capts(ifile, kw_file, standardize=False):
    data = pd.read_csv(ifile)
    img2capt = get_key2value_df(data, key='image_id', value='caption')
    keywords = list(pd.read_csv(kw_file)['keyword'])
    return extract_keywords_using_selected_list_multi_patterns(img2capt, keywords, standardize=standardize), img2capt


def save_extracted_keywords_to_file(img2keywords, img2capt, ofile):
    oids = []
    ocapts = []
    okeywords = []
    for img, ws in img2keywords.items():
        oids.append(img)
        okeywords.append(ws)
        ocapts.append(img2capt[img])

    df = pd.DataFrame({"image_id": oids, "caption": ocapts, "keywords": okeywords})
    df.to_csv(ofile)
    print("Saved to ", ofile)


def remove_sub_keywords(keywords):
    if len(keywords) == 0:
        return keywords
    b = [len(c) for c in keywords]
    indices = np.argsort(b)
    removed = []
    for i in range(len(keywords) - 1, 0, -1):
        idx_i = indices[i]
        if idx_i in removed:
            continue
        for j in range(i-1, -1, -1):
            idx_j = indices[j]
            if idx_j in removed:
                continue
            if keywords[idx_j] in keywords[idx_i]:
                removed.append(idx_j)
    filtered = [keywords[i] for i in range(len(keywords)) if i not in removed]
    return filtered


def rerun_for_empty_capt(keyword_file="data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_keywords_freq10_len3_manual.csv",
                         img2keyword_file="/Users/sonnguyen/Research/ReviewKG/data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_nouns_freq10_len3_manual.csv",
                         ofile="/Users/sonnguyen/Research/ReviewKG/data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_nouns_freq10_len3_manual_rerun.csv"):
    '''to deal with the case when only the keyword appear (^keyword$)'''
    keywords = list(pd.read_csv(keyword_file)['keyword'])
    data = pd.read_csv(img2keyword_file)
    data.keys()
    imgs = list(data['image_id'])
    capts = list(data['caption'])
    img_kws = list(data['keywords'])
    new_keywords = []
    canadd = []
    for i, img in tqdm(enumerate(imgs), total=len(imgs)):
        kws = ast.literal_eval(img_kws[i])
        if len(kws) == 0:
            capt = capts[i].lower().strip()
            if capt in keywords:
                canadd.append(img)
                new_keywords.append([capt])
            else:
                new_keywords.append([])
        else:
            new_keywords.append(kws)
    df = pd.DataFrame({"image_id": imgs, "caption": capts, "keywords": new_keywords})
    df.to_csv(ofile)
    print("Saved to", ofile)
    return canadd

nlp = spacy.load("en_core_web_sm")

####################################################################################################
# Step 1: extract NP using spacy
####################################################################################################
# analyze_file_customized('data/representation_learning/Image_Cap_Cat-WRIST_based/train.spacy.sim.csv',
#                         odir='data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized',
#                         lemma=True)

# --------other options-----------------------------------
# Option 1: take the original noun chunks  (will include DET, adj...)
# analyze_file('data/representation_learning/Image_Cap_Cat-WRIST_based/train.spacy.sim.csv', odir='data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_chunks', keep=None)

# Option 2: keep only ['NOUN', 'PROPN', 'VERB'] in noun chunks (failed to include some NPs)
analyze_file('data/representation_learning/Image_Cap_Cat-WRIST_based/test.spacy.sim.csv', odir='data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_chunks_NOUN_PROPN_VERB/', keep=['NOUN', 'PROPN', 'VERB'])

####################################################################################################
# Step 2: Construct a list of keywords (manual)
# 1. Sort keyword count by frequency
# 2. Check the coverage of reviews for a given number of top keywords (or a threshold)
# 3. Refine the keywords (having low frequency) --> match infrequent keywords to frequent keywords
# basil pesto pasta (4) --> pesto pasta (25)
# Ended up manually filter for freq 10 min-len 3
####################################################################################################
# np2count = get_key2value_df_file('data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train.spacy.sim_noun_count.csv', key='Keyword', value='Count')
# img2nps = get_key2value_df_file('data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train.spacy.sim_nouns.csv', key='image_id', value='keywords')
# np2c = tuple2dict(sorted(np2count.items(), key=lambda x:x[1], reverse=True))

# ----------------------------------------------------
# # check coverage (the number below including keywords with len less then 3, except for 40, 30, and 20)
# 1000    (#Keywords: 33)	0.20600237237058444	[44980/218347]
# 300 (#Keywords: 157)	0.37062565549332027	[80925/218347]
# 100 (#Keywords: 513)	0.5083926044323943	[111006/218347]
# 50  (#Keywords: 1039)	0.5861495692636034	[127984/218347]
# 40    (#Keywords: 1260)	0.6052338708569387	[132151/218347]
# 30    (#Keywords: 1636)	0.6320627258446417	[138009/218347]
# 20    (#Keywords: 2379)	0.6669338255162653	[145623/218347]
# 10*  (#Keywords: 4591)	0.7229776456740876	[157860/218347]
# 5   (#Keywords: 9105)	0.7764521610097689	[169536/218347]
# 3   (#Keywords: 16316)	0.8186189872084343	[178743/218347]
# ----------------------------------------------------
# check_coverage(img2nps, get_top_keywords(np2c, min_threshold=20, min_len=3))

# ----------------------------------------------------
# Save the selected keywords
# - Alternative, use excel to manually filter out unwanted keywords (do this)
# ----------------------------------------------------
# keywords = get_top_keywords(np2c, min_threshold=10, min_len=3, keep_count=True)  # 4533 keywords
# ofile = 'data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_selected_keywords_freq10_len3.json'
# json.dump(keywords, open(ofile, 'w'))

# ------- Check coverage: how many images having at least 1 keyword-----------------------------------
# check_empty_keywords('data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_nouns_all_refined_freq10_len3.csv')  # 0.9278762703403298 [202599/218347]  (Why? This used single pattern of all the keywords. Maybe because of the manually filtered keywords)
# check_empty_keywords('data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_nouns_freq10_len3_manual.csv')  # 0.8362881102098952 [182601/218347]
# check_empty_keywords('data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_nouns_freq10_len3_manual_no_subword.csv')  # 0.8722354783899023 [190450/218347]


####################################################################################################
# Below is draft

# ----- debug -------
# check single caption
# keywords = list(pd.read_csv("data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_keywords_freq10_len3_manual.csv")['keyword'])
# patterns = create_pattern(keywords, separate=True)
# for p in patterns:
#     if "lasagna" in p:
#         print(p)
# print(extract_keywords_using_selected_list_multi_patterns_for_an_image('Lasagna', patterns))
# patterns = create_pattern(keywords, separate=True)
# extract_keywords_using_selected_list_multi_patterns_for_an_image(img2capt['c9OTvWT5HNcfAaTlIjvlZg'], patterns=patterns)

# -----------re-run the non-keyword captions for the new template, for single word case----------
# Don't need to run this if re-run everything from the first step
# canadd = rerun_for_empty_capt()
# print(len(canadd))


# ----------compare nouns files-----------------
# f1 = pd.read_csv("data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_nouns_freq10_len3_manual.csv")
# f2 = pd.read_csv('data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/drafts/train_nouns_all_refined_freq10_len3.csv')
#
# d1 = get_key2value_df(f1, key='image_id', value='keywords', ast_value=True)
# d2 = get_key2value_df(f2, key='image_id', value='keywords', ast_value=True)
# i2c = get_key2value_df(f1, key='image_id', value='caption')
#
# tmp = []
# for i, k1 in d1.items():
#     k2 = d2[i]
#     if sorted(k1) != sorted(k2):
#         t = "{}\t{}\t{}\t{}".format(i, i2c[i], k1, k2)
#         tmp.append(t)
#
# pyperclip.copy('\n'.join(tmp))


# doc = nlp("I love Parties")
# for w in doc:
#     print(w, w.lemma_.lower())

# ======BEGIN========================
# ----- (not use) using single pattern for all keywords (not use because it failed to find 'chicken' for 'Top chicken makhani bottom paneer tikka masala')-------
# not use
# keywords = json.load(open('data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_selected_keywords_freq10_len3.json'))  # not use
# img2keywords = extract_keywords_using_selected_list(img2capt, img2nps, keywords, redo_all=True)
# +++++++DEBUG+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# 'Top chicken makhani bottom paneer tikka masala'
# - using the whole pattern (all keywords): ['tikka masala', 'top', 'bottom']
# - using only the part having chicken: pattern[:76] = ^(chicken)\W|\W(chicken)\W|\W(chicken)$|^(cheese)\W|\W(cheese)\W|\W(cheese)$ --> getting ['chicken']
# Maybe using too long pattern affects the quality of the search
# Solution, search for each keyword?
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# keywords = json.load(open('data/representation_learning/Image_Cap_Cat-WRIST_based_nouns/noun_phrases_customized_lemmatized/train_selected_keywords_freq10_len3.json'))
# img2capt = get_key2value_df(data, key='image_id', value='caption')
#
# pattern = create_pattern(keywords)
# print(get_matched_keywords(img2capt['c9OTvWT5HNcfAaTlIjvlZg'], pattern))
# print(get_matched_keywords(img2capt['c9OTvWT5HNcfAaTlIjvlZg'], pattern[:76]))
# print(pattern[:76])
# ======END========================

# # test debug
# pattern = create_pattern(keywords)
# sent = 'Top chicken makhani bottom paneer tikka masala'
# a = re.findall(pattern, sent.lower())
# a
# print(get_matched_keywords('Top chicken makhani bottom paneer tikka masala', pattern))
# pattern
#
# p = "^(chicken)\W|\W(chicken)\W|\W(chicken)$|^(cheese)\W|\W(cheese)\W|\W(cheese)$"
# b = re.findall(p, sent.lower())
# b

'''
- Choose: 
    NOUN; 
    ADJ?; 
    PROPN: California (California roll)
    VERB?: mashed potatoes (mashed: VERB); Chopped (VERB) brisket sandwich
- Not choose: 
    PRON: I, it, they
- 
'''
# test('Half a Mac and Cheese Burger')
#
# doc = nlp("Best Beef Wellington I've ever eaten.")
# for w in doc:
#     print(w, w.pos_)
# print()
# for chunk in doc.noun_chunks:
#     # print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
#     print(chunk.text)
#     for c in chunk:
#         print(c, c.pos_)
#     print("+++++\n")
#
# print(get_noun_phrases(doc))


# doc = nlp(add_space_after_dot("Trestles IPA by Left Coast Brewing.Cirrus, crisp and refreshingly smooth with 6.8% ABV and 58.5 IBU's"))
# for nc in doc.noun_chunks:
#     ws = []
#     print(nc)
#     for word in nc:
#         print(word, word.pos_)
#     print("+++")
# keep=['NOUN', 'PROPN', 'VERB']
# print(get_noun_phrases(doc, keep=keep))