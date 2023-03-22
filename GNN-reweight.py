import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from Dataset import MGTAB, Twibot20, Cresci15
from models import  GCN, SAGE, GAT, RGCN
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import sample_mask
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='MGTAB', choices=['MGTAB', 'Twibot20', 'Cresci15'], help='dataset')
parser.add_argument('-model', type=str, default='GCN', choices=['GCN', 'GAT', 'SAGE', 'RGCN'])
parser.add_argument('-reweight', type=str, default='CB', choices=['CB', 'FL'], help='CB loss, Focal loss')
parser.add_argument('--relation_select', type=list, default=[0,1], nargs='+', help='selection of relations in the graph (0-6)')
parser.add_argument('--random_seed', type=list, default=[0,1,2,3,4], nargs='+', help='selection of random seeds')
parser.add_argument('--hidden_dimension', type=int, default=128, help='number of hidden units')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate (1 - keep probability)')
parser.add_argument('--beta', type=float, default=0.9999, help='parameter for CB loss')
parser.add_argument('--gamma', type=float, default=2.0, help='parameter for reweight')
parser.add_argument('--alpha', type=float, default=0.5, help='parameter for FocaL loss')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def focal_loss(labels, logits, no_of_classes, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    labels_one_hot = F.one_hot(labels, no_of_classes).float()
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels_one_hot,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels_one_hot * logits - gamma * torch.log(1 +torch.exp(-1.0 * logits)))


    loss = modulator * BCLoss
    weighted_loss = torch.cat([alpha.unsqueeze(1), alpha.unsqueeze(1)], axis=1) * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= len(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss


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
    embedding_size = data.x.shape[1]
    relation_num = len(args.relation_select)
    index_select_list = (data.edge_type == 100)
    relation_dict = {
        0:'followers',
        1:'friends'
    }


    print('relation used:', end=' ')
    for features_index in args.relation_select:
            index_select_list = index_select_list + (features_index == data.edge_type)
            print('{}'.format(relation_dict[features_index]), end='  ')
    edge_index = data.edge_index[:, index_select_list]
    edge_type = data.edge_type[index_select_list]



    if args.model == 'RGCN':
        model = RGCN(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'GCN':
        model = GCN(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'GAT':
        model = GAT(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'SAGE':
        model = SAGE(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)

    def train(epoch):
        model.train()
        output = model(data.x, edge_index, edge_type)
        beta = args.beta
        gamma = args.gamma
        samples_per_cls = [len(data.y[data.train_mask]) - sum(data.y[data.train_mask]).cpu().item(),
                           sum(data.y[data.test_mask]).cpu().item()]
        if args.alpha is not None:
            ma_cls = samples_per_cls.index(max(samples_per_cls))
            mi_cls = samples_per_cls.index(min(samples_per_cls))
            alpha = (data.y[data.train_mask] == ma_cls)*args.alpha + (data.y[data.train_mask] == mi_cls)
        else:
            alpha = sum(samples_per_cls) / samples_per_cls[-1] * data.y[data.train_mask] + (1 - data.y[data.train_mask])
        loss_type = "softmax"
        no_of_classes = len(samples_per_cls)
        if args.reweight == 'CB':
            loss_train = CB_loss(data.y[data.train_mask], output[data.train_mask], samples_per_cls, no_of_classes, loss_type, beta, gamma)
        else:
            loss_train = focal_loss(data.y[data.train_mask], output[data.train_mask], no_of_classes, alpha, gamma)

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
        output = model(data.x, edge_index, edge_type)
        beta = args.beta
        gamma = args.gamma

        samples_per_cls = [len(data.y[data.test_mask]) - sum(data.y[data.test_mask]).cpu().item(),
                           sum(data.y[data.test_mask]).cpu().item()]
        if args.alpha is not None:
            ma_cls = samples_per_cls.index(max(samples_per_cls))
            mi_cls = samples_per_cls.index(min(samples_per_cls))
            alpha = (data.y[data.test_mask] == ma_cls) * args.alpha + (data.y[data.test_mask] == mi_cls)
        else:
            alpha = sum(samples_per_cls) / samples_per_cls[-1] * data.y[data.test_mask] + (1 - data.y[data.test_mask])
        loss_type = "softmax"
        no_of_classes = len(samples_per_cls)
        if args.reweight == 'CB':
            loss_test = CB_loss(data.y[data.test_mask], output[data.test_mask], samples_per_cls, no_of_classes, loss_type,
                            beta, gamma)
        else:
            loss_test = focal_loss(data.y[data.test_mask], output[data.test_mask], no_of_classes, alpha, gamma)
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
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    acc_class0_list = []
    acc_class1_list = []
    bacc_list = []

    for i, seed in enumerate(args.random_seed):
        print('traning {}th model\n'.format(i + 1))
        acc, precision, recall, f1, acc_class0, acc_class1 = main(seed)
        acc_list.append(acc * 100)
        precision_list.append(precision * 100)
        recall_list.append(recall * 100)
        f1_list.append(f1 * 100)
        acc_class0_list.append(acc_class0 * 100)
        acc_class1_list.append(acc_class1 * 100)
        bacc_list.append((acc_class0 + acc_class1) * 50)

    print('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    print('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    print('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    print('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list)))
    print('acc_class0:{:.2f} + {:.2f}'.format(np.array(acc_class0_list).mean(), np.std(acc_class0_list)))
    print('acc_class1:{:.2f} + {:.2f}'.format(np.array(acc_class1_list).mean(), np.std(acc_class1_list)))
    print('bAcc:      {:.2f} + {:.2f}'.format(np.array(bacc_list).mean(), np.std(bacc_list)))
    print('total time:', time.time() - t)




