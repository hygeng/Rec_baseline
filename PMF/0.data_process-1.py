import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
# choose dataset to process
dataset = 'ml-100k'
raw_data_path = os.path.join(os.getcwd(), 'data', dataset, 'ratings.dat')
processed_data_path = os.path.join(os.getcwd(), 'processed_data', dataset)

names = ['user_id', 'item_id', 'rating', 'timestamp']
read = pd.read_csv(raw_data_path, sep='\t', names=names,engine = 'python')



users = list(read['user_id'].unique())
user_id_index = dict((user_id, index) for user_id, index in zip(users, range(len(users))))
items = list(read['item_id'].unique())
item_id_index = dict((item_id, index) for item_id, index in zip(items, range(len(items))))
print("start reading")
# data = []
# with open(os.getcwd()+'/data/'+dataset+'/ratings.dat', 'r') as f:
#     lines = f.readlines()
#     count_user = 0
#     count_item = 0
#     for i in tqdm(range(len(lines))):
#         line = lines[i].strip().split('\t')
#         user_id = int(line[0])
#         item_id = int(line[1])
#         rating = float(line[2])

#         data.append([user_id_index[user_id], item_id_index[item_id], rating])



read['user_id'] = read['user_id'].map(user_id_index )
read['item_id'] = read['item_id'].map(user_id_index )

read = read.dropna(axis=0,how='any') #drop all rows that have any NaN values
read.reset_index(drop = True)

read['user_id'] = read['user_id'].astype('int')
read['item_id'] = read['item_id'].astype('int')
data = read[['user_id', 'item_id', 'rating']].values

# data = np.array(data)
np.random.shuffle(data)

pickle.dump(user_id_index, open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'wb'))
pickle.dump(item_id_index, open(os.path.join(processed_data_path, 'item_id_index.pkl'), 'wb'))
# np.savetxt(os.path.join(processed_data_path, 'data.txt'), data, fmt='%f')
np.save(os.path.join(processed_data_path, 'data.npy'), data)