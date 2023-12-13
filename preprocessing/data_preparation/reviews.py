'''
convert reviews to csv file 
'''

import os 
import pandas as pd 
import json 
from tqdm import tqdm 


def create_reversed_list(k2v):
    '''
    return v2k 
    '''
    v2k = {}
    for k, v in k2v.items():
        for vid in v: 
            v2k[vid] = k
    return v2k 


def get_review_info(review, rid, bid, city, setname=None):
    tmp = {'review_id': rid, "user_id": review['user_id'], 'city': city, "rest_id": bid, 
            'photos': get_photo_ids(review['photos']), 'date': review['date'], "rating": review['rating'], "text": review['text'].replace("\r", " ").replace("\n", " ")}
    if setname is not None: 
        tmp['setname'] = setname
    return tmp 


def get_photo_ids(photos):
    tmp = []
    for p in photos:
        tmp.append(p['img_id'])
    return tmp 

def combine_reviews_with_split(include_none=True):
    rdir = '/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users/'
    if include_none:
        ofile = 'data/input/reviews_all_split_with_none_uid.csv'
    else:
        ofile = 'data/input/reviews_all_split.csv'
    splits = json.load(open('data/input/users_split.json'))
    all_reviews = [] 
    # data = {'train': [], "val": [], "test": []}
    # txt = 'All the goodness of the deep South! Seasoned and pan bronzed. Mississippi catfish served over Gouda '
    for city in os.listdir(rdir):
        if city.startswith('.'):
            continue 
        # if city != 'lasvegas':
            # continue 
        print("Processing for", city)
        idir = os.path.join(rdir, city, 'extracted_reviews')
        split_uid2set = create_reversed_list(splits[city])
        tmp = os.listdir(idir)
        for fname in tqdm(tmp, total=len(tmp)):
            if fname.startswith("."):
                continue 
            bid = fname[:-5]
            ifile = os.path.join(idir, fname)
            dt = json.load(open(ifile))
            for k, v in dt.items():
                setname = split_uid2set.get(v['user_id'], None)
                if not include_none and setname is None:
                    continue 
                rv = get_review_info(v, k, bid, city, setname)
                all_reviews.append(rv)
    df = pd.DataFrame(all_reviews)
    print("#reviews: {}".format(len(all_reviews)))
    print("#df: {}".format(len(df)))
    df.to_csv(ofile)
    print("Saved to", ofile)


def combine_reviews_no_split(allow_none_user=False):
    rdir = '/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users/'
    if allow_none_user:
        ofile = 'data/input/reviews_all_with_none_uid.csv'
    else:
        ofile = 'data/input/reviews_all.csv'
    all_reviews = [] 
    # data = {'train': [], "val": [], "test": []}
    for city in os.listdir(rdir):
        if city.startswith('.'):
            continue 
        # if city != 'lasvegas':
            # continue 
        print("Processing for", city)
        idir = os.path.join(rdir, city, 'extracted_reviews')
        tmp = os.listdir(idir)
        for fname in tqdm(tmp, total=len(tmp)):
            if fname.startswith("."):
                continue 
            bid = fname[:-5]
            ifile = os.path.join(idir, fname)
            dt = json.load(open(ifile))
            for k, v in dt.items():
                if not allow_none_user and v['user_id'] is None:
                    continue 
                rv = get_review_info(v, k, bid, city)
                all_reviews.append(rv)
    df = pd.DataFrame(all_reviews)
    print("#reviews: {}".format(len(all_reviews)))
    print("#df: {}".format(len(df)))
    df.to_csv(ofile)
    print("Saved to", ofile)


# combine_reviews_with_split(include_none=False)  # with setname information 
combine_reviews_no_split(allow_none_user=False)  # no setname info 

