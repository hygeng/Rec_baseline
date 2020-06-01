import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.train_dir + args.dataset):
    os.makedirs(args.train_dir + args.dataset)
# with open(os.path.join(args.train_dir + args.dataset, 'args.txt'), 'w') as f:
print('\n'.join([str(k) + ',' + str(v) for k, v in sorted(list(vars(args).items()), key=lambda x: x[0])]))
# f.close()
t0 = time.time()
dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = int(len(user_train) / args.batch_size)
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))
print("load data:",time.time()-t0)
f = open(os.path.join(args.train_dir + args.dataset, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())

T = 0.0
t0 = time.time()


for epoch in range(1, args.num_epochs + 1):
    # for step in tqdm(list(range(num_batch)), total=num_batch, ncols=70, leave=False, unit='b'):
    for step in range(num_batch):
        u, seq, pos, neg = sampler.next_batch()
        auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                    model.is_training: True})
    print(".",end = "")
    if epoch % 10 == 0:
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end=' ')
        t_test = evaluate(model, dataset, args, sess)
        t_valid = evaluate_valid(model, dataset, args, sess)
        print('')
        print('epoch:%d, time: %f(s), valid (NDCG@10: %.6f, HR@10: %.6f, Recall@10: %.6f), test (NDCG@10: %.6f, HR@10: %.6f, Recall@10: %.6f)' % (
        epoch, T, t_valid[0], t_valid[1], t_valid[2], t_test[0], t_test[1], t_test[2]))

        print(str(t_valid) + ' ' + str(t_test) + '\n')
        # f.flush()
        t0 = time.time()


# f.close()
sampler.close()
print("Done")
