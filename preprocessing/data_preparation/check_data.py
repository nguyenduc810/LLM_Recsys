'''
check if reviews_all file contains all reviews
# 1684726: #reviews from '/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users'
# 1675684: #reviews from data/input/reviews_all.csv
# reason: reviews_all removed reviews with None user 
'''

import os 
import json
from tqdm import tqdm 
import pandas as pd 


idir = '/Users/nguyents/Library/CloudStorage/OneDrive-ASTAR/workspace/datasets/yelp-ts-cleaned/reviews_users'
counter = 0
rids = []
for city in os.listdir(idir):
    if city.startswith('.'):
        continue
    rdir = os.path.join(idir, city, 'extracted_reviews')
    for rfile in tqdm(os.listdir(rdir)):
        if rfile.startswith("."):
            continue
        dt = json.load(open(os.path.join(rdir, rfile)))
        counter += len(dt)
        rids += list(dt.keys())

print("#reviews: {}".format(counter))

# review file 
rfile = 'data/input/reviews_all.csv'
data = pd.read_csv(rfile)
len(data)
data.keys()
rids2 = data['review_id']

missing = list(set(rids) - set(rids2))
len(missing)

