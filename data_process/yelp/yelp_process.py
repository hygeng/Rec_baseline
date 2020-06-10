import pandas as pd
import json
import csv
import numpy as np
import joblib
from joblib import Parallel, delayed
import datetime
import copy
from copy import deepcopy
import math
import scipy
from scipy import spatial
import scipy.spatial
# import tensorflow as tf
import time
import glob
import os,sys
import subprocess
import dill
from tqdm import tqdm
import pickle

import collections

import pandas as pd
import numpy as np
from tqdm import tqdm
import json,os,sys
import numpy as np

# from scipy.interpolate import spline
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


######################################################################################################
################################### user2id, item2id  ############################################
######################################################################################################
def id_dict():
    df = pd.read_csv("/home/hygeng/data/yelp/review.csv",index_col = 0)
    print("read done")
    unique_sid = pd.unique(df['business_id'])
    unique_uid = pd.unique(df['user_id'])
    item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    user2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    with open('/home/hygeng/data/yelp/analysis/user_dict.json','w') as outfile:
        json.dump(user2id,outfile,ensure_ascii=False)
        outfile.write('\n')
    print("user done")
    with open('/home/hygeng/data/yelp/analysis/item_dict.json','w') as outfile:
        json.dump(item2id,outfile,ensure_ascii=False)
        outfile.write('\n')
    print("item done")

    # json_file = open("/home/hygeng/data/yelp/analysis/user_dict.json", 'r')
    # user2id = json.load(json_file)

    # json_file = open("/home/hygeng/data/yelp/analysis/item_dict.json",'r')
    # item2id  =json.load(json_file)
    
    # df['business_id'] = df['business_id'].apply(lambda x: item2id[x])
    # df['user_id'] = df['user_id'].apply(lambda x: user2id[x])

######################################################################################################
################################### meta data processing  ############################################
######################################################################################################

def yelp_meta_process():
    fname = "/home/hygeng/data/yelp/review.json"
    state_file = "/home/hygeng/data/yelp/analysis/state_dict.json"
    json_file = open(state_file, 'r')
    state_dict = json.load(json_file)
    
    out_path = "/home/hygeng/data/yelp/all.txt"
    f_out = open(out_path, "w")
    with open(fname, 'r',encoding = "utf-8") as f:
        print("open done")
        lines = f.readlines()
        print("read done")
        i = 0

        for line in tqdm(lines):
        #if (i > 100 ):
        #   break
            # if (i % 10000 == 0):
            #     print(str(i) + "/" + str(len(lines)))
            i += 1
           # if i >20:
            #    break
            tmp = json.loads(json.dumps(eval(line)))
            
            try:
                id = tmp['user_id']
                item = tmp['business_id']
                star = tmp['stars']
                date = tmp['date']
                st = state_dict[item]

              #  print(tmp['unixReviewTime'])
                f_out.write(tmp['user_id'])
                f_out.write('\t')
                f_out.write(tmp['business_id'])
                f_out.write('\t')
                f_out.write(str(tmp['stars']))
                f_out.write('\t')
                f_out.write(str(tmp['date']) + '\n')

            except:
                continue
            
    f_out.close()

def produce_csv():
    txt_file = open("all.txt","r")
    users = []
    items = []
    ratings = []
    times = []
    
    for line in tqdm(txt_file.readlines()):
        linelist = line.strip().split('\t')
        # print(linelist)
        users.append(linelist[0])
        items.append(linelist[1])
        ratings.append(linelist[2])
        times.append(linelist[3])
        # continue
    csv_dict = {"user":users, "item": items, "rating":ratings,"time":times}
    df = pd.DataFrame(csv_dict)
    df.head()
    df.to_csv("all.csv")

def one_hot(min_uc=50,min_sc=200,issave=False):
    # produce one-hot encoding for user and item, for NFM
    raw_data = pd.read_csv("all.csv")
    # Filter Data
    raw_data = raw_data[raw_data['rating'] > 3.5]
    raw_data, user_activity, item_popularity = filter_triplets(raw_data,min_uc,min_sc)

    # Shuffle User Indices
    unique_uid = user_activity.index
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    # print("num_user:",n_users)
    n_heldout_users = int(n_users*0.1)
    

    # Split Train/Validation/Test User Indices
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['item'])

    print("unique_uid:",unique_uid.shape)
    print("unique_sid:",unique_sid.shape)
    # assert 0

    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    show2id = dict((sid, i+unique_uid.shape[0]) for (i, sid) in enumerate(unique_sid))

    np.save('one_hot_user.npy', profile2id) 
    np.save('one_hot_item.npy', show2id) 

    DATA_DIR = "/home/hygeng/data/yelp/archive/"
    pro_dir =  os.path.join(DATA_DIR, "nfm_"+str(min_uc)+'_'+str(min_sc)+'/pro_sg')

    if not os.path.exists(pro_dir) and issave:
        os.makedirs(pro_dir)
    
    if issave:
        with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)
    
    if not issave:
        f_out = open("out.txt","a+")
        print("min_uc:",min_uc,"min_sc:",min_sc,file = f_out)
    print("min_uc:",min_uc,"min_sc:",min_sc)
    

    vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_sid)]
    
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
    # print("valid done!")

    test_plays = raw_data.loc[raw_data['user'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['item'].isin(unique_sid)]\

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)
    # print("test done!")
    
    if not issave:
        print("user:",user_activity.shape,"item:",item_popularity.shape,file = f_out)
    else:
        print("user:",user_activity.shape,"item:",item_popularity.shape)

    train_data = numerize_nfm(train_plays, profile2id, show2id)
    if issave:
        train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
        print("train:",train_data.shape)
    else:
        print("train:",train_data.shape,file = f_out)

    vad_data_tr = numerize_nfm(vad_plays_tr, profile2id, show2id)
    if issave:
        vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
        print("vad_data_tr:",vad_data_tr.shape)
    else:
        print("vad_data_tr:",vad_data_tr.shape,file = f_out)

    vad_data_te = numerize_nfm(vad_plays_te, profile2id, show2id)
    if issave:
        vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
        print("vad_data_te:",vad_data_te.shape)
    else:
        print("vad_data_te:",vad_data_te.shape,file = f_out)

    test_data_tr = numerize_nfm(test_plays_tr, profile2id, show2id)
    if issave:
        test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
        print("test_data_tr:",test_data_tr.shape)
    else:
        print("test_data_tr:",test_data_tr.shape,file = f_out)

    test_data_te = numerize_nfm(test_plays_te, profile2id, show2id)
    if issave:
        test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
        print("test_data_te:",test_data_te.shape)
    else:
        print("test_data_te:",test_data_te.shape,file = f_out)
        print("\n\n",file = f_out)

    print("Done!")

def dataset_split_vae_cf(filename, DATA_DIR, quartile_num =0, min_uc=70,min_sc=70,issave=False):
# filename = "/home/hygeng/data/yelp/quartile/AZ/5/ratings_after_filter.csv"
# min_uc=5
# min_sc=5
# issave=True

    raw_data = pd.read_csv(filename)
    # print("data shape:",raw_data.shape)
    # Filter Data
    # raw_data = raw_data[raw_data['rating'] > 3.5]
    raw_data, user_activity, item_popularity = filter_triplets(raw_data,min_uc,min_sc)


    # assert 0
    # Shuffle User Indices
    unique_uid = user_activity.index
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    # print("num_user:",n_users)
    n_heldout_users = int(n_users*0.1)
    # assert 0

    # Split Train/Validation/Test User Indices
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]

    # print("train,valid, test",tr_users.shape,vd_users.shape, te_users.shape)

    train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['item'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))


    pro_dir =  os.path.join(DATA_DIR, str(min_uc)+'_'+str(min_sc)+'/'+str(quartile_num)+'/pro_sg')

    if not os.path.exists(pro_dir) and issave:
        os.makedirs(pro_dir)
    
    if issave:
        with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)
    
    if not issave:
        f_out = open("out.txt","a+")
        print("min_uc:",min_uc,"min_sc:",min_sc,file = f_out)
    print("min_uc:",min_uc,"min_sc:",min_sc)
    


    vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_sid)]
    
    # print("vad_data_tr:",vad_plays.shape)

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
    # print("valid done!")
    

    test_plays = raw_data.loc[raw_data['user'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['item'].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)
    # print("test done!")

    if not issave:
        print("entries:",raw_data.shape, "user:",user_activity.shape,"item:",item_popularity.shape, "sparsity=", float(raw_data.shape[0])/(user_activity.shape[0]*item_popularity.shape[0]), file = f_out)
    print("entries:",raw_data.shape, "user:",user_activity.shape,"item:", item_popularity.shape, "sparsity=", float(raw_data.shape[0])/(user_activity.shape[0]*item_popularity.shape[0]))


    train_data = numerize(train_plays, profile2id, show2id)
    if issave:
        train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
        print("train:",train_data.shape)
    else:
        print("train:",train_data.shape,file = f_out)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    if issave:
        vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
        print("vad_data_tr:",vad_data_tr.shape)
    else:
        print("vad_data_tr:",vad_data_tr.shape,file = f_out)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    if issave:
        vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
        print("vad_data_te:",vad_data_te.shape)
    else:
        print("vad_data_te:",vad_data_te.shape,file = f_out)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    if issave:
        test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
        print("test_data_tr:",test_data_tr.shape)
    else:
        print("test_data_tr:",test_data_tr.shape,file = f_out)

    test_data_te = numerize(test_plays_te, profile2id, show2id)
    if issave:
        test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
        print("test_data_te:",test_data_te.shape)
    else:
        print("test_data_te:",test_data_te.shape,file = f_out)
        print("\n\n",file = f_out)

    print("Done!")


def produce_csv_with_state():
    state_file = "/home/hygeng/data/yelp/analysis/state_dict.json"
    json_file = open(state_file, 'r')
    state_dict = json.load(json_file)

    csv_file = "/home/hygeng/data/yelp/all.csv"
    raw_data = pd.read_csv(csv_file)

    df = pd.DataFrame(raw_data)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['state'] = df['item'].apply(lambda x: state_dict[x])
    df.to_csv("/home/hygeng/data/yelp/ratings_with_state.csv")

######################################################################################################
################################### data processing utils  ############################################
######################################################################################################
def numerize(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def numerize_nfm(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])

    # print(uid.iloc[0])
    for i in tqdm(range(uid.shape[0])):
        uid.iloc[i] = str(uid.iloc[i]) + ':1'
    for i in tqdm(range(sid.shape[0])):
        sid.iloc[i] = str(sid.iloc[i]) + ':1'
    return pd.DataFrame(data={'label':"1", 'uid': uid, 'sid': sid}, columns=['label','uid', 'sid'])


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=5, min_sc=5):
    def filter_user(tp, min_uc=5):
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]
        return tp
    def filter_item(tp, min_sc=5):
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]
        return tp
    tp = filter_user(tp, min_uc)
    tp = filter_item(tp, min_sc)
    tp = filter_user(tp, min_uc)
    tp = filter_item(tp, min_sc)
    tp = filter_user(tp, min_uc)
    tp = filter_item(tp, min_sc)
    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):
    # print("datashape:",data.shape)
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()
    # print("start")
    np.random.seed(98765)
    i = 0
    for _, group in tqdm(data_grouped_by_user):
        n_items_u = len(group)
        # if i % 10000 == 0:
        #     print(str(i) + "/" + str(len(data_grouped_by_user)))
        i += 1
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)
    #print(tr_list)
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    return data_tr, data_te

def positive_samples(filename,data_dir,min_uc =5, min_sc=5):
    # filename = "/home/hygeng/data/yelp/quartile/AZ/5/ratings_after_filter.csv"
    # data_dir = "/home/hygeng/data/yelp/archive/trans/AZ/5_5/"
    # min_uc=5
    # min_sc=5
    # issave=True

    raw_data = pd.read_csv(filename)
    raw_data, user_activity, item_popularity = filter_triplets(raw_data,min_uc,min_sc)
    unique_sid = pd.unique(raw_data['item'])
    unique_uid = pd.unique(raw_data['user'])
    item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    user2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    raw_data['item'] = raw_data['item'].apply(lambda x: item2id[x])
    raw_data['user'] = raw_data['user'].apply(lambda x: user2id[x])

    pos_samples = {}

    data_grouped_by_user = raw_data.groupby('user')
    for user, group in tqdm(data_grouped_by_user):
        pos_samples[user]= list(group.loc[:,'item'])
        
    with open(data_dir+'/pos_samples.json','w') as outfile:
        json.dump(pos_samples,outfile,ensure_ascii=False)
        outfile.write('\n')

######################################################################################################
################################### data analysis      ################################################
######################################################################################################

def calculate_user_interaction():
    raw_data = pd.read_csv("all.csv")
    # Filter Data
    # raw_data = raw_data[raw_data['rating'] > 3.5]

    data_grouped_by_user = raw_data.groupby('user')
    
    # total = 0
    dist = np.array([])
    for _, group in tqdm(data_grouped_by_user):
        n_items_u = len(group)
        dist = np.hstack((dist,n_items_u))
        # total += n_items_u
    # print(float(total)/len(data_grouped_by_user))
    np.save("/home/hygeng/data/yelp/analysis/user_interactions.npy",dist)


def statistics_produce_csv():
    txt_file = open("/home/hygeng/data/yelp/all.csv", "r")#open("../NCF/Data/ml-1m.train.rating","r")
    users = []
    items = []
    ratings = []
    times = []
    iid = []
    idx = 0
    nn = 0
    for line in tqdm(txt_file.readlines()):
        if nn == 0:
            nn += 1
            continue
        linelist = line.strip().split(',')
        # print(linelist)
        # assert 0
        users.append(linelist[1])
        items.append(linelist[2])
        ratings.append(linelist[3])
        times.append(linelist[4])
        iid.append(idx)
        idx += 1
        # continue
    csv_dict = {"iid": iid, "user":users, "item": items, "rating":ratings,"time":times}
    df = pd.DataFrame(csv_dict)
    df.head()
    df.to_csv("/home/hygeng/data/yelp/statistics_all.csv")


def quartile_split(user_threshold = 5,state_name = None, save_dir="/home/hygeng/data/yelp/quartile/"):
# user_threshold = 5
# state_name = 'AZ' 
# save_dir="/home/hygeng/data/yelp/quartile/"

    data = pd.read_csv("/home/hygeng/data/yelp/ratings_with_state.csv", sep=',', header=0, names=['user', 'item','rating','time','state'], usecols=[1,2, 3,4,5])#, dtype={0:np.int32,1:np.int32, 2:np.int32})
    print("all data shape:", data.shape)
    if state_name!=None:
        data = data[data['state'] == state_name]
# data_grouped_by_user = data.groupby('user')

    data, usercount, itemcount = filter_triplets(data, min_uc=5, min_sc=5)
    user_rank = get_count(data,'user').reset_index(name='count').sort_values(by=['count'])
    user_list = user_rank[user_rank['count'] >= user_threshold].set_index('user')
    # print(data_list.iloc[0])
    # print(len(data_grouped_by_user["iid"]))

    prefix_dir = save_dir+str(state_name)+"/"+str(user_threshold)+"/"
    if not os.path.exists(prefix_dir) :
        os.makedirs(prefix_dir)
    data.to_csv(prefix_dir+"ratings_after_filter.csv")

    user_list['quartile_num'] = 0
    quartile_user_num = int(len(user_list) / 4)
    user_list.iloc[:quartile_user_num,1]=1
    user_list.iloc[quartile_user_num:2* quartile_user_num,1]=2
    user_list.iloc[quartile_user_num *2: quartile_user_num*3,1]=3
    user_list.iloc[quartile_user_num*3:,1]=4

    user_dict = user_list.set_index(user_list.index)['quartile_num'].T.to_dict()
    # user_dict = user_list['quartile_num'].to_dict('list')
    with open('/home/hygeng/data/yelp/analysis/user_quartile_dict.json','w') as outfile:
        json.dump(user_dict,outfile,ensure_ascii=False)
        outfile.write('\n')

    print( quartile_user_num)
    quartile = [user_list[user_list['quartile_num']==i]['count'] for i in range(1,5)]
    for i in range(4):
        print("quartile %s:"%(i+1), "size:",len(quartile[i]),"min:", quartile[i].min(),"max:",quartile[i].max(), "average", float(sum(quartile[i]))/ quartile_user_num)

def quartile_save(user_threshold = 5,state_name = 'AZ',save_dir="/home/hygeng/data/yelp/quartile/"):
    data = pd.read_csv("/home/hygeng/data/yelp/quartile/"+str(state_name)+"/"+str(user_threshold)+"/ratings_after_filter.csv", sep=',', header=0, names=['user', 'item','rating','time','state'], usecols=[1,2, 3,4,5])

    filename = '/home/hygeng/data/yelp/analysis/user_quartile_dict.json'
    json_file = open(filename, 'r')
    user_dict = json.load(json_file)

    data['quartile_num'] =data['user'].map(user_dict)


    if state_name ==None:
        prefix_dir = save_dir+str(user_threshold)
    else:
        prefix_dir = save_dir+state_name+"/"+str(user_threshold)

    if not os.path.exists(prefix_dir) :
        os.makedirs(prefix_dir)
    print("start save!",flush=True)
    data[data['quartile_num']==1].to_csv(prefix_dir+"/quartile_1.csv")
    data[data['quartile_num']==2].to_csv(prefix_dir+"/quartile_2.csv")
    data[data['quartile_num']==3].to_csv(prefix_dir+"/quartile_3.csv")
    data[data['quartile_num']==4].to_csv(prefix_dir+"/quartile_4.csv")

def state_partition():
    #get dictionary of state and restaurant category
    fname = '/home/hygeng/data/yelp/business.json'

    # out_path = "all.txt"
    state_dict={}
    category_dict = {}
    # f_out = open(out_path, "w")
    with open(fname, 'r',encoding = "utf-8") as f:
        print("open done")
        lines = f.readlines()
        print("read done")
        i = 0

        for line in tqdm(lines):
            # if (i % 10000 == 0):
            #     print(str(i) + "/" + str(len(lines)))
            i += 1
           # if i >20:
            #    break
            
            try:
                tmp = json.loads(json.dumps(eval(line)))
                idx = tmp['business_id']
                st = tmp['state']
                ca = tmp['categories']

                state_dict[idx] = st
                category_dict[idx] = ca

            except:
                continue
            
    # f_out.close()
    with open('/home/hygeng/data/yelp/analysis/state_dict.json','w') as outfile:
        json.dump(state_dict,outfile,ensure_ascii=False)
        outfile.write('\n')

    with open('/home/hygeng/data/yelp/analysis/category_dict.json','w') as outfile:
        json.dump(category_dict,outfile,ensure_ascii=False)
        outfile.write('\n')
    print("Done!")

######################################################################################################
################################### analysis Visualization ###########################################
######################################################################################################


def visual():
    num = np.load("/home/hygeng/data/yelp/analysis/user_interactions.npy")
    num = np.sort(num)
    maximum = int(np.max(num))
    dist = [np.log(np.sum(num==i)) for i in tqdm(range(1,maximum))]
    x = [np.log(i) for i in range(1,maximum)]


    plt.title("Yelp user_interactions log-log graph")
    plt.xlabel("log(interactions)")
    plt.ylabel("log(users)")
    # print(dist)
    plt.plot(x,dist)
    # plt.show()
    plt.savefig("/home/hygeng/data/yelp/analysis/user_interactions_loglog.png")


def visual_state_distribution():
    filename = '/home/hygeng/data/yelp/analysis/state_dict.json'
    json_file = open(filename, 'r')

    state_dict = json.load(json_file)
    state_distri = pd.DataFrame(state_dict.values())
    print(len(state_dict.values()))
    result = state_distri.apply(pd.value_counts)
    print(result)
    # assert 0

def visual_category_distribution():
    filename = '/home/hygeng/data/yelp/analysis/category_dict.json'
    json_file = open(filename, 'r')

    cate_dict = json.load(json_file)
    dict_values = list(cate_dict.values())
    print(dict_values[0].split(','))
    all_cate = []
    for item in tqdm(dict_values):
        for k in item.split(','):
            all_cate.append(k.strip()) 
    cate_distri = pd.DataFrame(all_cate)
    print(len(cate_distri))
    result = cate_distri.apply(pd.value_counts)
    result.columns = ["count"]
    print(result.columns)
    # file_out.close()
    result.to_csv('/home/hygeng/data/yelp/analysis/category.csv', sep=',', header=True, index=True)

    plt.title("Yelp business categorey")
    plt.xlabel("category")
    plt.ylabel("number of business")
    # print(dist)
    plt.ylim(0, 9000)  
    plt.bar(x= range(result.shape[0]), height = result['count'])
    # plt.show()
    plt.savefig("/home/hygeng/data/yelp/analysis/category.png")


######################################################################################################
################################### context/seq part  ############################################
######################################################################################################

def dataset_split_translation(filename, data_dir, quartile_num =0, min_uc=5,min_sc=5):
    # filename = "/home/hygeng/data/yelp/quartile/AZ/5/ratings_after_filter.csv"
    # data_dir = "/home/hygeng/data/yelp/archive/trans/AZ/5_5/"
    # min_uc=5
    # min_sc=5
    # issave=True

    raw_data = pd.read_csv(filename)
    # print("data shape:",raw_data.shape)
    # Filter Data
    # raw_data = raw_data[raw_data['rating'] > 3.5]
    raw_data, user_activity, item_popularity = filter_triplets(raw_data,min_uc,min_sc)
    unique_sid = pd.unique(raw_data['item'])
    unique_uid = pd.unique(raw_data['user'])
    item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    user2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    raw_data['item'] = raw_data['item'].apply(lambda x: item2id[x])
    raw_data['user'] = raw_data['user'].apply(lambda x: user2id[x])

    f = open(data_dir + "unique_uid.txt")
    for sid, i in item2id:
        f.write(sid)
        f.write("\t")
        f.write(i)
    f.close()

    f = open(data_dir + "unique_sid.txt")
    for uid, i in user2id:
        f.write(uid)
        f.write("\t")
        f.write(i)
    f.close()

    user_train = {}
    user_val = {}
    user_test = {}
    usernum = len(user2id)
    itemnum = len(item2id)

    data_grouped_by_user = raw_data.groupby('user')
    for user, group in tqdm(data_grouped_by_user):
        user_time_rank = group.sort_values(by=['time'])
        user_test[user] = user_time_rank.iloc[-1,2] # [i,i'] 
        user_val[user] = [user_time_rank.iloc[-3,2],user_time_rank.iloc[-2,2]] # [i,i']
        user_train[user] = list(user_time_rank.iloc[:-3,2]) # [i,i']
        # user_test[user] = [user_time_rank.iloc[-3,2],user_time_rank.iloc[-2,2],user_time_rank.iloc[-1,2]] # [i,i']
        # user_val[user] = [user_time_rank.iloc[-4,2],user_time_rank.iloc[-3,2]] # [i,i']
        # user_train[user] = list(user_time_rank.iloc[0:-2,2]) # [i,i']

    # data_dir = DATA_DIR
    np.save(data_dir+"transrec_data.npy", [user_train,user_val,user_test,usernum,itemnum])
    # [user_train,user_val,user_test,usernum,itemnum] = da

def see_context():
    fname = "/home/hygeng/data/yelp/business.json"
    cate_set = set()
    with open(fname, 'r',encoding = "utf-8") as f:
        lines = f.readlines()
        print("read done")
        for line in tqdm(lines):
            try:
                tmp = json.loads(json.dumps(eval(line)))
                id = tmp['categories']
                for cate in id:
                    cate_set.add(cate)
            except:
                continue
    print(len(cate_set))
######################################################################################################
################################### Main function   ##################################################
######################################################################################################
def main():
    if  len(sys.argv) !=3:
        print("error! use python yelp_process $min_uc $min_sc")
    else:
        # yelp_meta_process()
        # produce_csv()
        # filename = "/home/hygeng/data/yelp/all.csv"
        filename = "/home/hygeng/data/yelp/quartile/AZ/5/ratings_after_filter.csv"
        DATA_DIR = "/home/hygeng/data/yelp/archive/AZ/"
        dataset_split_vae_cf(filename,DATA_DIR,quartile_num =0, min_uc=int(sys.argv[1]),min_sc=int(sys.argv[2]),issave=True)
        # one_hot(min_uc=int(sys.argv[1]),min_sc=int(sys.argv[2]),issave=True)

def quartile_main():
    if  len(sys.argv) !=3:
        print("error! use python yelp_process $min_uc $min_sc")
    else:
        # yelp_meta_process()
        # produce_csv()
        for num in range(1,5):
            filename = "/home/hygeng/data/yelp/quartile/AZ/5/quartile_"+str(num)+".csv"
            DATA_DIR = "/home/hygeng/data/yelp/quartile/AZ/5"
            dataset_split_vae_cf(filename,DATA_DIR,quartile_num =num, min_uc=0,min_sc=0,issave=True)
        # one_hot(min_uc=int(sys.argv[1]),min_sc=int(sys.argv[2]),issave=True)


# main()



#data analysis
# calculate()
# visual()


# quartile split workflow:
# quartile_split(user_threshold = 5, state_name = 'AZ',save_dir="/home/hygeng/data/yelp/quartile/")
# quartile_save(user_threshold = 5,state_name = 'AZ',save_dir="/home/hygeng/data/yelp/quartile/")
# quartile_main()


# state_partition()
# visual_state_distribution()
# visual_category_distribution()

# see_context()

# filename = "/home/hygeng/data/yelp/quartile/AZ/5/ratings_after_filter.csv"
# data_dir = "/home/hygeng/data/yelp/archive/AZ/5_5/"
# dataset_split_translation(filename, data_dir, quartile_num =0, min_uc=5, min_sc=5)

filename = "/home/hygeng/data/yelp/quartile/AZ/5/ratings_after_filter.csv"
data_dir = "/home/hygeng/data/yelp/seq_archive/AZ/5_5/"
positive_samples(filename,data_dir,5,5)

#################################### user friends #################################
with open("yelp_dataset/user.json", 'r') as f:
    counter = 0
    with open("user.csv",'w') as csv_f:
        writer = csv.DictWriter(csv_f, delimiter=',', fieldnames=['user_id','friends'],quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for line in f:
            data = json.loads(line)
            # print data
            writer.writerow({'user_id':data["user_id"],'friends':data['friends']})
            counter+=1


##########################################  load csv  ################################
df_business = pd.read_csv('business_ghy.csv',names=['business_id','city','state','latitude','longitude','review_count','attr_list','categories','hours'],header=0)
df_review = pd.read_csv('review.csv',names=['user_id','business_id','stars','text','date'],header=0)
df_user = pd.read_csv('user.csv',names=['user_id','friends'],header=0)
df_photo = pd.read_csv('photo.csv',names = ['photo_id','business_id','caption','label'],header=0)

##########################################  filter the states and create dict  ################################
df_statewise = df_business.groupby('state')
df_business = df_business.groupby('state').filter(lambda x: len(x) >= 1000)
df_business.state.unique()

business_by_state = {}
for state in df_business.state.unique():
    business_by_state[state] = df_business[df_business['state']==state]

review_by_state = {}
for state in business_by_state:
    print(state)
    review_by_state[state] = df_review.merge(business_by_state[state][['business_id','city','categories']],how='inner',left_on='business_id',right_on='business_id')
    
##########################################  filter user and item, save data  ################################
#filter the user and item

users_per_item = 5
items_per_user = 5
review_by_state_filtered = {}
for state in review_by_state:
    print(state)
    prev_len = len(review_by_state[state])
    x = review_by_state[state].groupby('business_id').filter(lambda x: len(x) >= users_per_item)
    next_len = len(x)
    while (next_len < prev_len):
        print(next_len, prev_len)
        prev_len = next_len
        x = x.groupby('user_id').filter(lambda x: len(x) >= items_per_user)
        x = x.groupby('business_id').filter(lambda x: len(x) >= users_per_item)
        next_len = len(x)
    review_by_state_filtered[state] = x

##########################################  user2id item2id  ################################
# unique_sid = pd.unique(df['business_id'])
# unique_uid = pd.unique(df['user_id'])
# item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
# user2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
# with open('/home/hygeng/data/yelp/analysis/user_dict.json','w') as outfile:
#     json.dump(user2id,outfile,ensure_ascii=False)
#     outfile.write('\n')
# print("user done")
# with open('/home/hygeng/data/yelp/analysis/item_dict.json','w') as outfile:
#     json.dump(item2id,outfile,ensure_ascii=False)
#     outfile.write('\n')
# print("item done")


json_file = open("/cluster/home/it_stu141/data/yelp/seq/context/user_dict.json", 'r') 
user2id = json.load(json_file) 
json_file = open("/cluster/home/it_stu141/data/yelp/seq/context/item_dict.json",'r') 
item2id =json.load(json_file)

state = 'NC'
df =  review_by_state_filtered[state]
df = df.merge(df_user[['user_id','friends']],how='left',left_on='user_id',right_on='user_id')
df['business_id'] = df['business_id'].apply(lambda x: item2id[x])
df['user_id'] = df['user_id'].apply(lambda x: user2id[x])

unique_photo_id = df_photo['photo_id'].unique()
photo2id = dict((pid, i) for (i, pid) in enumerate(unique_photo_id))
id2photo =  dict([(v,k) for (k,v) in photo2id.items()])

##########################################  id redirect  ################################
user_redirect = dict((uid, i) for (i, uid) in enumerate(np.sort(df['user_id'].unique())))
item_redirect = dict((sid, i) for (i, sid) in enumerate(np.sort(df['business_id'].unique())))

df['business_id'] = df['business_id'].apply(lambda x: item_redirect[x])
df['user_id'] = df['user_id'].apply(lambda x: user_redirect[x])
##########################################  get data  ################################


df_all = collections.defaultdict(set)
df_train = collections.defaultdict(set)
df_val = collections.defaultdict(set)
df_test = collections.defaultdict(set)

data_grouped_by_user = df.groupby('user_id')
for user, group in tqdm(data_grouped_by_user):
    # print(group.ix[-1].index.values)
    user_time_rank = group.sort_values(by=['date'])
    # print(user,user_time_rank)
    df_all[user] = user_time_rank.iloc[:,1].values
    df_train[user] = user_time_rank.iloc[:-2,1].values
    df_val[user] = user_time_rank.iloc[-2,1]
    df_test[user] = user_time_rank.iloc[-1,1]
    # assert 0

data_dir =  "/cluster/home/it_stu141/data/yelp/HASC/"
np.save(data_dir + "user_ratings_all.npy",df_all)
np.save(data_dir + "user_ratings_train.npy",df_train)
np.save(data_dir + "user_ratings_val.npy",df_val)
np.save(data_dir + "user_ratings_test.npy",df_test)

# for index, row in tqdm(df.iterrows()):
#     # print(row['user_id'],row['friends']) 
#     # assert 0
#     friends = []
#     for fri in row['friends'].split(', '):
#         try:
#             friends.append(str(user_redirect[user2id[fri]]))
#         except:
#             continue
#     df.loc[index,'friends_id'] = ','.join(friends)

user_follows = []
data_grouped_by_user = df.groupby('user_id')
for user, group in tqdm(data_grouped_by_user):
    # print(group.loc[:,'friends'].values)
    # assert 0
    friends = set([i for friends in group.loc[:,'friends'].values for i in friends.split(', ') ])
    friends_id =  []
    for fri in friends:
        try:
            friends_id.append(user_redirect[user2id[fri]]) #
        except:
            continue
    user_follows.append(friends_id)
user_follows = np.array(user_follows)
data_dir =  "/cluster/home/it_stu141/data/yelp/HASC/"
np.save(data_dir + "user_follows.npy",user_follows)


df_photo['business_id'] = df_photo['business_id'].apply(lambda x: item2id[x])

shop_image_dict = {}
data_grouped_by_item = df_photo.groupby('business_id')
for item, group in tqdm(data_grouped_by_item):
    try:
        shop_image_dict[item_redirect[item]]=group.iloc[:,0].values
    except:
        continue



# user_ups
user_ups = []
for user in tqdm(df_train):
    user_up = []
    for item in df_train[user]:
        if item not in shop_image_dict:
            continue
        for image in shop_image_dict[item]:
            try:
                user_up.append(photo2id[image])
            except:
                continue
    user_ups.append(sorted(user_up))
user_ups = np.array(user_ups)
np.save(data_dir+"user_ups.npy",user_ups)


from PIL import Image



photo_dir = "/cluster/home/it_stu141/data/yelp/yelp_photos/photos/"
img_feature = np.zeros(200000*64*64).reshape(200000,64,64)
for idx in tqdm(range(len(id2photo))):
    try:
        name = photo_dir + id2photo[idx] + ".jpg"
        img = np.array(Image.open(name).resize((64,64)).convert('L'), 'f')
        img_feature[idx,:,:] = img
    except:
        continue
        # print(idx)
np.save(data_dir + "img_feature.npy",img_feature)

photo_dir = "/cluster/home/it_stu141/data/yelp/yelp_photos/photos/"
photo_names = os.listdir(photo_dir) 


