import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, datetime , os

from tensorboardX import SummaryWriter
from scipy import sparse
import models
import data
import metric

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='/home/chenchao/data/SGMC-ICML22/',
                    help='dataset location')
parser.add_argument('--dataname', type=str, default='Cora', help='dataname')               
parser.add_argument('--mask_his', action='store_true', default=False)
parser.add_argument('--start_from_1', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=250,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device',type=str, default="3",
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--topk', type=int, default=10,
                    help='topk')
args = parser.parse_args()

# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     if not args.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# device = torch.device("cuda" if args.cuda else "cpu")
device = torch.device("cuda:{}".format(args.device) if (torch.cuda.is_available() and args.device!="cpu") else "cpu")
print("using device:", device)
data_dir = "{}/{}/".format(args.data, args.dataname)
print("data path: ", data_dir)
print("mask history: ", args.mask_his)
###############################################################################
# Load data
###############################################################################

loader = data.DataLoader(data_dir, args.start_from_1)

train_data = loader.load_data('train')
vad_data_tr, vad_data_te = loader.load_data('validation')
test_data_tr, test_data_te = loader.load_data('test')
n_items = loader.load_n_items()

N = train_data.shape[0]
idxlist = list(range(N))

###############################################################################
# Build the model
###############################################################################

p_dims = [200, 600, n_items]
# p_dims = [50, n_items]
model = models.MultiVAE(p_dims).to(device)

lr_val = args.lr
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = models.loss_function

###############################################################################
# Training code
###############################################################################

# TensorboardX Writer
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
logdir_ = './runs/%s/%s' % (args.dataname, now)
if not os.path.exists('%s/' % logdir_):
    os.makedirs('%s/' % logdir_)
print("save to:", logdir_)
writer = SummaryWriter(logdir = logdir_)
# file writing
f1 = open('{}/logs.txt'.format(logdir_),'w')

def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(np.nan_to_num(data.toarray()))


def train():
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = np.nan_to_num(train_data[idxlist[start_idx:end_idx]])
        data = naive_sparse2tensor(data).to(device)

        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 
                            1. * update_count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        loss = criterion(recon_batch, data, mu, logvar, anneal)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            sys.stdout.flush()
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format( epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval, train_loss / args.log_interval))

            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format( epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval, train_loss / args.log_interval), file=f1)
            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(range(0, N, args.batch_size)) + batch_idx
            writer.add_scalars('data/loss', {'train': train_loss / args.log_interval}, n_iter)

            start_time = time.time()
            train_loss = 0.0


def evaluate(data_tr, data_te):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    HR_list = np.array([])
    PR_list = np.array([])
    NDCG_list = np.array([])
    RC_list = np.array([])
    F1_list = np.array([])
    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, e_N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 
                               1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data_tensor)

            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            heldout_data = heldout_data.A
            
            if args.mask_his:
                recon_batch[data.nonzero()] = -np.inf
            # recon_batch[np.where(heldout_data== 0)] = -np.inf
            # print(recon_batch.shape, heldout_data.shape)

            HR, PR,  NDCG, RC, F1 = metric.evaluate(recon_batch, heldout_data, topk=args.topk)
            
            HR_list = np.hstack((HR_list, HR))
            PR_list = np.hstack((PR_list, PR))
            NDCG_list = np.hstack((NDCG_list, NDCG))
            RC_list = np.hstack((RC_list, RC))
            F1_list = np.hstack((F1_list, F1))


    total_loss /= len(range(0, e_N, args.batch_size))
    # print(n10_list)

    return total_loss, np.mean(HR_list), np.mean(PR_list), np.mean(NDCG_list), np.mean(RC_list), np.mean(F1_list)


best_model = -np.inf
update_count = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss, HR, PR,  NDCG, RC, F1 = evaluate(vad_data_tr, vad_data_te)
        sys.stdout.flush()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:.5f}s | valid loss {:.5f} | '
                'HR {:.5f} | PR {:.5f} | NDCG {:.5f} | RC  {:.5f} | F1  {:.5f} '.format(
                    epoch, time.time() - epoch_start_time, val_loss, HR, PR,  NDCG, RC, F1))
        print('| end of epoch {:3d} | time: {:.5f}s | valid loss {:.5f} | '
                'HR {:.5f} | PR {:.5f} | NDCG {:.5f} | RC  {:.5f} | F1  {:.5f} '.format(
                    epoch, time.time() - epoch_start_time, val_loss, HR, PR,  NDCG, RC, F1), file = f1)
        print('-' * 89)

        n_iter = epoch * len(range(0, N, args.batch_size))
        writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
        writer.add_scalar('data/HR', HR, n_iter)
        writer.add_scalar('data/PR', PR, n_iter)
        writer.add_scalar('data/NDCG', NDCG, n_iter)
        writer.add_scalar('data/RC', RC, n_iter)
        writer.add_scalar('data/F1', F1, n_iter)


        # Save the model if the n10 is the best we've seen so far.
        if HR > best_model:
            with open(logdir_ + "/model.pt", 'wb') as f:
                torch.save(model, f)
            best_model = HR

except KeyboardInterrupt:
    sys.stdout.flush()
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(logdir_ + "/model.pt", 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss, HR, PR,  NDCG, RC, F1= evaluate(test_data_tr, test_data_te)
sys.stdout.flush()
print('=' * 89)
print('| End of training | test loss {:.5f} | HR {:.5f} | PR {:.5f} | NDCG{:.5f} | RC  {:.5f} | F1  {:.5f} '.format(
            test_loss,  HR, PR,  NDCG, RC, F1))
print('| End of training | test loss {:.5f} | HR {:.5f} | PR {:.5f} | NDCG{:.5f} | RC  {:.5f} | F1  {:.5f} '.format(
            test_loss,  HR, PR,  NDCG, RC, F1), file=f1)
print('=' * 89)
print("log dir: ", logdir_)
f1.close()