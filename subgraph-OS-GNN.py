import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from Dataset import MGTAB, Twibot20, Cresci15
from models import RGCN, GAT, GCN, SAGE
from models import SMOTERGCN, SMOTEGAT, SMOTEGCN, SMOTESAGE
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import normalize, sparse_mx_to_torch_sparse_tensor, sample_mask, MLSMOTE, balance_MLSMOTE
from collections import Counter
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, choices=['MGTAB', 'Twibot20', 'Cresci15'], help='dataset')
parser.add_argument('-model', type=str, default='GAT', help='GCN, GAT, SAGE, RGCN')
parser.add_argument('-smote', type=bool, default=True, help='whether use smoteGCN')
parser.add_argument('-ratio', type=float, default=0.05, help='imbalanced ratio')
parser.add_argument('--relation_select', type=list, default=[0,1], nargs='+', help='selection of relations in the graph (0-6)')
parser.add_argument('--random_seed', type=list, default=[0,1,2,4,5], nargs='+', help='selection of random seeds')
parser.add_argument('--balanced', type=bool, default=True, help='whether use balanced smote')
parser.add_argument('--smote_num', type=int, default=300, help='number of minority class samples after synthesis, only works when balanced is set to False')
parser.add_argument('--hidden_dimension', type=int, default=128, help='number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (1 - keep probability)')
parser.add_argument('--alpha', type=float, default=0.8, help='weight of synthesized samples')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)


def main(seed):

    if args.dataset == 'MGTAB':
        dataset = MGTAB('./Data/MGTAB')
        data = dataset[0]
        data.y = data.y2
    elif args.dataset == 'Twibot20':
        dataset = Twibot20('./Data/Twibot20')
        data = dataset[0]
    else:
        dataset = Cresci15('./Data/Cresci15')
        data = dataset[0]

    out_dim = max(data.y).item()+1
    sample_number = len(data.y)

    num_count = []
    for i in range(out_dim):
        num_count.append(list(data.y.numpy()).count(i))

    ma_cls = num_count.index(max(num_count))
    mi_cls = num_count.index(min(num_count))

    ma_mask = data.y == ma_cls
    mi_mask = data.y == mi_cls
    all_idx = np.array(range(len(data.y)))
    ma_idx = all_idx[ma_mask]
    shuffle_mi_idx = shuffle(all_idx[mi_mask], random_state=seed)

    mi_idx = shuffle_mi_idx[:int(args.ratio*len(ma_idx))]
    mi_mask = sample_mask(mi_idx, sample_number)
    data.y = data.y[ma_mask + mi_mask]
    data.x = data.x[ma_mask + mi_mask]
    mi_label_mask = data.y == mi_cls
    ma_label_mask = data.y == ma_cls
    data.y[ma_label_mask] = 0
    data.y[mi_label_mask] = 1
    select_idx = all_idx[ma_mask + mi_mask]

    edge_select = []
    for i in range(data.edge_index.shape[1]):
        edge_select.append(data.edge_index[0, i].item() in select_idx and data.edge_index[1, i].item() in select_idx)

    data.edge_index = data.edge_index[:, edge_select]

    select_idx2idx = {index: i for i, index in enumerate(select_idx)}
    src = torch.tensor([select_idx2idx[index.item()] for index in data.edge_index[0, :]])
    dst = torch.tensor([select_idx2idx[index.item()] for index in data.edge_index[1, :]])
    data.edge_type = data.edge_type[edge_select]
    data.edge_index = torch.cat([src.unsqueeze(dim=1), dst.unsqueeze(dim=1)], axis=1)
    data.edge_index = data.edge_index.t().contiguous()

    sample_number = len(data.y)
    shuffled_idx = shuffle(np.array(range(sample_number)), random_state=seed)
    train_idx = shuffled_idx[:int(0.1 * sample_number)]
    val_idx = shuffled_idx[int(0.1 * sample_number):int(0.2 * sample_number)]
    test_idx = shuffled_idx[int(0.2 * sample_number):]

    data.train_mask = sample_mask(train_idx, sample_number)
    data.val_mask = sample_mask(val_idx, sample_number)
    data.test_mask = sample_mask(test_idx, sample_number)

    test_mask = data.test_mask
    train_mask = data.train_mask
    val_mask = data.val_mask

    data = data.to(device)
    relation_num = len(args.relation_select)
    index_select_list = (data.edge_type == 100)

    relation_dict = {
        0:'followers',
        1:'friends'
    }
    edge_index = data.edge_index
    edge_type = None
    features = data.x
    labels = data.y
    embedding_size = features.shape[1]

    if args.smote:
        boosting_features = features
        for features_index in args.relation_select:
            if isinstance(features_index, list):
                for index in features_index:
                    index_select_list = index_select_list + (index == data.edge_type)
                    print('{}'.format(relation_dict[index]), end='  ')
            else:
                index_select_list = index_select_list + (features_index == data.edge_type)
                print('{}'.format(relation_dict[features_index]), end='  ')
            edge_index = data.edge_index[:, index_select_list]
            edge_type = data.edge_type[index_select_list]

            adj = sp.coo_matrix((np.ones(edge_index.cpu().shape[1]), (edge_index.cpu()[0, :], edge_index.cpu()[1, :])),
                                shape=(features.cpu().shape[0], features.cpu().shape[0]),
                                dtype=np.float32)

            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = normalize(adj + sp.eye(adj.shape[0]))
            adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

            super_once_nodes = torch.spmm(adj, features)
            super_twice_nodes = torch.spmm(adj, super_once_nodes)
            boosting_features = torch.cat([boosting_features, super_twice_nodes], axis=1)

        smote_embedding_size = boosting_features.shape[1]
        labeled_X = boosting_features[train_idx, :].cpu().numpy()
        all_X = boosting_features.cpu().numpy()
        labeled_y = labels[train_idx].cpu().numpy()
        all_y = labels.cpu().numpy()
        print(Counter(labeled_y))

        num_count = []
        for i in range(max(data.y) + 1):
            num_count.append(list(data.y[train_mask].cpu().numpy()).count(i))
        args.smote_num = max(num_count)

        if args.balanced:
            X_smo, y_smo = balance_MLSMOTE(labeled_X, labeled_y, args.smote_num)
        else:
            X_smo, y_smo = MLSMOTE(labeled_X, labeled_y, args.smote_num)
        print(Counter(y_smo))

        idx_except_train = torch.LongTensor(range(len(labels)))[~data.train_mask]
        orign_idx_train = torch.tensor(np.array(range(len(train_idx))), dtype=torch.long).cuda()

        new_idx_train = torch.tensor(np.array(range(len(y_smo))), dtype=torch.long).cuda()
        new_idx_val = torch.tensor(val_idx, dtype=torch.long).cuda() + torch.tensor(len(y_smo) - len(train_idx)).cuda()
        new_idx_test = torch.tensor(test_idx, dtype=torch.long).cuda() + torch.tensor(len(y_smo) - len(train_idx)).cuda()

        X_generate = torch.FloatTensor(np.concatenate([X_smo, all_X[idx_except_train, :]], axis=0)).cuda()
        y_generate = torch.LongTensor(np.concatenate([y_smo, all_y[idx_except_train]], axis=0)).cuda()
    else:
        print('relation used:', end=' ')
        for features_index in args.relation_select:
            index_select_list = index_select_list + (features_index == data.edge_type)
            print('{}'.format(relation_dict[features_index]), end='  ')

        edge_index = data.edge_index[:, index_select_list]
        edge_type = data.edge_type[index_select_list]


    if args.model == 'RGCN':
        model = RGCN(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
        if args.smote:
            SMOTEmodel = SMOTERGCN(smote_embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'GCN':
        model = GCN(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
        if args.smote:
            SMOTEmodel = SMOTEGCN(smote_embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'GAT':
        model = GAT(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
        if args.smote:
            SMOTEmodel = SMOTEGAT(smote_embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'SAGE':
        model = SAGE(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
        if args.smote:
            SMOTEmodel = SMOTESAGE(smote_embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)


    if args.smote:
        model = SMOTEmodel
        print('After oversampling', Counter(y_smo))


    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)


    def train(epoch):
        model.train()
        if args.smote:
            output = model(X_generate, edge_index, edge_type)
            alpha = args.alpha
            loss_train = alpha * loss(output[new_idx_train], y_generate[new_idx_train]) + (1 - alpha) * loss(
                output[orign_idx_train], y_generate[orign_idx_train])
            out = output.max(1)[1]
            acc_train = accuracy_score(out[new_idx_train].to('cpu'), y_generate[new_idx_train].to('cpu'))
            acc_val = accuracy_score(out[new_idx_val].to('cpu'), y_generate[new_idx_val].to('cpu'))
        else:
            output = model(data.x, edge_index, edge_type)
            loss_train = loss(output[data.train_mask], labels[data.train_mask])
            out = output.max(1)[1].to('cpu').detach().numpy()
            label = data.y.to('cpu').detach().numpy()
            acc_train = accuracy_score(out[train_mask], label[train_mask])
            acc_val = accuracy_score(out[val_mask], label[val_mask])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if (epoch + 1)%50 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()), )
        return acc_val


    def test():
        model.eval()
        if args.smote:
            output = model(X_generate, edge_index, edge_type)
            loss_test = loss(output[new_idx_test], y_generate[new_idx_test])
            out = output.max(1)[1]
            acc_test = accuracy_score(out[new_idx_test].to('cpu'), y_generate[new_idx_test].to('cpu'))
            f1 = f1_score(out[new_idx_test].to('cpu'), y_generate[new_idx_test].to('cpu'), average='macro')
            precision = precision_score(out[new_idx_test].to('cpu'), y_generate[new_idx_test].to('cpu'), average='macro')
            recall = recall_score(out[new_idx_test].to('cpu'), y_generate[new_idx_test].to('cpu'), average='macro')
            mask_class0 = (y_generate[new_idx_test] == 0)
            mask_class1 = (y_generate[new_idx_test] == 1)
            acc_test_class0 = accuracy_score(out[new_idx_test][mask_class0].to('cpu'), y_generate[new_idx_test][mask_class0].to('cpu'))
            acc_test_class1 = accuracy_score(out[new_idx_test][mask_class1].to('cpu'), y_generate[new_idx_test][mask_class1].to('cpu'))
        else:
            output = model(data.x, edge_index, edge_type)
            loss_test = loss(output[data.test_mask], labels[data.test_mask])
            out = output.max(1)[1].to('cpu').detach().numpy()
            label = data.y.to('cpu').detach().numpy()
            acc_test = accuracy_score(out[test_mask], label[test_mask])
            f1 = f1_score(out[test_mask], label[test_mask], average='macro')
            precision = precision_score(out[test_mask], label[test_mask], average='macro')
            recall = recall_score(out[test_mask], label[test_mask], average='macro')
            mask_class0 = (label[test_mask] == 0)
            mask_class1 = (label[test_mask] == 1)
            acc_test_class0 = accuracy_score(out[test_mask][mask_class0], label[test_mask][mask_class0])
            acc_test_class1 = accuracy_score(out[test_mask][mask_class1], label[test_mask][mask_class1])
        return acc_test, loss_test, f1, precision, recall, acc_test_class0, acc_test_class1

    model.apply(init_weights)

    epochs = 200
    max_val_acc = 0
    for epoch in range(epochs):
        acc_val = train(epoch)
        acc_test, loss_test, f1, precision, recall, acc_test_class0, acc_test_class1 = test()
        if acc_val > max_val_acc:
            max_val_acc = acc_val
            max_acc = acc_test
            max_epoch = epoch + 1
            max_f1 = f1
            max_precision = precision
            max_recall = recall
            max_acc_test_class0 = acc_test_class0
            max_acc_test_class1 = acc_test_class1

    print("Test set results:",
          "epoch= {:}".format(max_epoch),
          "acc= {:.4f}".format(max_acc),
          "precision= {:.4f}".format(max_precision),
          "recall= {:.4f}".format(max_recall),
          "f1= {:.4f}".format(max_f1),
          "acc_class0= {:.4f}".format(max_acc_test_class0),
          "acc_class1= {:.4f}".format(max_acc_test_class1)
          )
    return max_acc, max_precision, max_recall, max_f1, max_acc_test_class0, max_acc_test_class1


if __name__ == "__main__":

    t = time.time()
    acc_list =[]
    precision_list = []
    recall_list = []
    f1_list = []
    acc_class0_list = []
    acc_class1_list = []
    bacc_list = []

    for i, seed in enumerate(args.random_seed):
        print('traning {}th model\n'.format(i+1))
        acc, precision, recall, f1, acc_class0, acc_class1 = main(seed)
        acc_list.append(acc*100)
        precision_list.append(precision*100)
        recall_list.append(recall*100)
        f1_list.append(f1*100)
        acc_class0_list.append(acc_class0*100)
        acc_class1_list.append(acc_class1*100)
        bacc_list.append((acc_class0 + acc_class1) * 50)

    print('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    print('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    print('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    print('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list)))
    print('acc_class0:{:.2f} + {:.2f}'.format(np.array(acc_class0_list).mean(), np.std(acc_class0_list)))
    print('acc_class1:{:.2f} + {:.2f}'.format(np.array(acc_class1_list).mean(), np.std(acc_class1_list)))
    print('bAcc:      {:.2f} + {:.2f}'.format(np.array(bacc_list).mean(), np.std(bacc_list)))
    print('total time:', time.time() - t)