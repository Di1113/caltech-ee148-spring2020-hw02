import numpy as np
import os
import json

np.random.seed(2020) # to ensure you always get the same train/test split

data_path = '../../data/RedLights2011_Medium'
gts_path = '../../data/hw02_annotations'
split_path = '../../data/hw02_splits'
os.makedirs(split_path, exist_ok=True) # create directory if needed

split_test = True # set to True and run when annotations are available

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test
file_names_train = []
file_names_test = []
'''
Your code below. 
'''

np.random.shuffle(file_names)
train_size = int(train_frac*len(file_names))
file_names_train = file_names[:train_size]
file_names_test = file_names[train_size :]
# print(len(file_names_train))
# print(len(file_names_test))

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train,file_names_test)) == 0

np.save(os.path.join(split_path,'file_names_train.npy'),file_names_train)
np.save(os.path.join(split_path,'file_names_test.npy'),file_names_test)

if split_test:
    with open(os.path.join(gts_path, 'formatted_annotations_mturk.json'),'r') as f:
        gts = json.load(f)
        # print(json.dumps(gts, sort_keys=True, indent=4))
    
    # Use file_names_train and file_names_test to apply the split to the
    # annotations
    gts_train = {}
    gts_test = {}
    '''
    Your code below. 
    '''
    # convert list to set for faster lookup 
    test_set = set(file_names_test)
    for key in gts:
        if key in test_set:
            gts_test[key] = gts.get(key)
        else: 
            gts_train[key] = gts.get(key)
        
    # print(json.dumps(gts_train, sort_keys=True, indent=4))
    # print(json.dumps(gts_test, sort_keys=True, indent=4))
    # print(len(gts_train))
    # print(len(gts_test)) 

    with open(os.path.join(gts_path, 'annotations_train.json'),'w') as f:
        json.dump(gts_train,f)
    
    with open(os.path.join(gts_path, 'annotations_test.json'),'w') as f:
        json.dump(gts_test,f)
    
    
