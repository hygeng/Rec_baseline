import os,random
import pandas as pd
from scipy import sparse
import numpy as np
from tqdm import tqdm
import json

class DataLoader():
    '''
    Load Movielens-20m dataset
    '''
    def __init__(self, path, start_from_1=False):
        self.pro_dir = path #os.path.join(path, 'pro_sg')
        assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"
        self.start_from_1 = start_from_1
        # self.n_items = self.load_n_items()
    
    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")
        
    def load_n_items(self):
        # unique_sid = list()
        # with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
        #     for line in f:
        #         unique_sid.append(line.strip())
        # n_items = 3706#len(unique_sid)
        return self.n_items
    
    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')
        
        tp = pd.read_csv(path)
        if self.start_from_1:
            tp['sid'] = tp['sid'] - 1

        n_users = tp['uid'].max() + 1
        # print("item min and max: ", tp['sid'].max(), tp['sid'].min())
        self.n_items = tp['sid'].max() - tp['sid'].min() + 1
        rows, cols = tp['uid'], tp['sid']

        # negative sampling part:
        # min_sid = tp['sid'].min()
        # max_sid = tp['sid'].max()
        # neg_samples_row  = np.array(rows)
        # neg_samples_col  = np.array(cols)
        # for uid, group in tp.groupby('uid'):
        #     # print(group[0])
        #     pos_samples = group['sid'].values
        #     neg_pool = [sid for sid in range(min_sid,max_sid) if sid not in pos_samples]
        #     neg_samples = random.sample(neg_pool , min(99*group.shape[0],len(neg_pool)))
        #     neg_samples_row = np.hstack((neg_samples_row, [uid for _ in range(len(neg_samples))]))
        #     neg_samples_col = np.hstack((neg_samples_col,neg_samples))
        # neg_samples_label = np.hstack((np.ones_like(rows),np.zeros(neg_samples_col.shape[0]-rows.shape[0])))
        # end

        data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, self.n_items))
        # data = sparse.csr_matrix((neg_samples_label, (neg_samples_row, neg_samples_col)), dtype='float64', shape=(n_users, self.n_items))
        return data
    
    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        # sid_path = os.path.join(self.pro_dir,'unique_sid.txt')
        # unique_sid = open(sid_path,'r').readlines()
        # show2id = dict((sid.strip(), i) for (i, sid) in enumerate(unique_sid))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        min_sid = tp_te['sid'].min()
        max_sid = tp_te['sid'].max()

        if self.start_from_1:
            tp_te['sid'] = tp_te['sid'] - 1
            tp_tr['sid'] = tp_tr['sid'] - 1
        
        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        # negative sampling part:
        # neg_samples_row  = np.array(rows_te)
        # neg_samples_col  = np.array(cols_te)
        # user_group = tp_te.groupby('uid')

        # pos_samples = user_group.head(1).values
        # neg_samples_row = np.array(pos_samples[:,0]-start_idx)
        # neg_samples_col = np.array(pos_samples[:,1])

        # for uid, group in user_group:
            # print(group.iloc[0,0], group.iloc[0,1])
            # assert 0
            # pos_samples = group['sid'].values
            # neg_pool = [sid for sid in range(min_sid,max_sid) if sid not in pos_samples]
            # neg_samples = random.sample(neg_pool , 99) # min(99*group.shape[0],len(neg_pool)))
            # neg_samples_row = np.hstack((neg_samples_row, [uid-start_idx for _ in range(99)]))
            # neg_samples_col = np.hstack((neg_samples_col,neg_samples))
        # neg_samples_label = np.hstack((np.ones_like(rows_te),np.zeros(neg_samples_col.shape[0]-rows_te.shape[0])))
        # neg_samples_label = np.hstack((np.ones(len(user_group)),-1*np.ones(len(user_group)*99)))
        # end

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        # data_te = sparse.csr_matrix((neg_samples_label,(neg_samples_row, neg_samples_col)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te
    

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]
    
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]
    
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        
        else:
            tr_list.append(group)
        
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    uid = tp['userId'].apply(lambda x: profile2id[x])
    sid = tp['movieId'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])



if __name__ == '__main__':

    print("Load and Preprocess Movielens-20m dataset")
    # Load Data
    DATA_DIR = '/home/hygeng/data/movielens/ml-1m/'
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)
    raw_data = raw_data[raw_data['rating'] > 3.5]

    # Filter Data
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)

    # Shuffle User Indices
    unique_uid = user_activity.index
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    n_heldout_users = int(n_users*0.1)

    # Split Train/Validation/Test User Indices
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['movieId'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(DATA_DIR, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)
    
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)
    
    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

    print("Done!")