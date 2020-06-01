import sys
# sys.path.append("/..") 
sys.path.append(sys.path[0]+"/..")

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os
import pickle

df_path = "/cluster/home/it_stu110/data/ciao/Ciao/done/ciao.csv"
save_dir = "/cluster/home/it_stu110/proj/baseline/SAMN/data/ciao/"



def get_friends(df,num_users):
    df_u = df[['user_id','friends']].drop_duplicates()
    df_u = df_u.reset_index(drop = True)
    df_u = df_u.fillna("")
    # df_u['friends'] = df_u['friends'].apply(lambda x: [int(fri) for fri in x.split(', ')] if x!="" else [])
    df_u.drop(columns = ['user_id'])
    friends_dict = df_u.drop(columns = ['user_id'])['friends'].apply(lambda x: [int(fri) for fri in str(x).split(", ")] if str(x)!="" else [] ).to_dict() 
    maxlen = 0
    for idx in range(df_u.shape[0]):
        maxlen  = max(maxlen,len(friends_dict[idx]))
        if ([int(fri) for fri in df_u.loc[idx,'friends'].split(', ')] if df_u.loc[idx,'friends']!="" else []) != friends_dict[idx]:
            print(idx)
    for idx in tqdm(range(df_u.shape[0])):
        if len(friends_dict[idx])<maxlen:
            friends_dict[idx].extend([num_users-1 for _ in range(maxlen - len(friends_dict[idx]))])
        if(len(friends_dict[idx])!=maxlen):
            print(idx)

    with open(save_dir + "trust.dic", "wb") as fp:   #Pickling
        pickle.dump(friends_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)      

def data_split(df):
    df_sample = df[["user_id","business_id"]].sample(frac=1.0)
    df_sample.columns = ["uid","sid"]
    df_sample = df_sample.reset_index(drop = True)
    shape = df_sample.shape[0]
    df_train = df_sample.loc[0:shape*0.7]
    df_valid = df_sample.loc[shape*0.7:shape*0.9]
    df_test = df_sample.loc[shape*0.9:]

    df_train  = df_train.sort_values("sid").sort_values("uid")
    df_valid  = df_valid.sort_values("sid").sort_values("uid")
    df_test  = df_test.sort_values("sid").sort_values("uid")

    df_train.to_csv(save_dir + "train.csv",index=False)
    df_valid.to_csv(save_dir + "valid.csv",index=False)
    df_test.to_csv(save_dir + "test.csv",index=False)

def uid_sid(df):
    unique_uid = df['user_id'].unique()
    unique_sid = df['business_id'].unique()

    ufile = open(save_dir + 'unique_uid_sub.txt', 'w')
    for uid in unique_uid :
        ufile.write(str(uid))
        ufile.write('\n')
    # 
    ufile.write(str(uid+1))
    ufile.write('\n')
    ufile.close()

    sfile = open(save_dir + 'unique_sid_sub.txt', 'w')
    for sid in unique_sid :
        sfile.write(str(sid))
        sfile.write('\n')
    sfile.close()
    return len(unique_uid)+1, len(unique_sid)

def pipeline():
    df = pd.read_csv(df_path,index_col = 0)
    print("read done")
    num_users, num_items = uid_sid(df)
    print("uid_sid done")
    data_split(df)
    get_friends(df,num_users)

pipeline()





