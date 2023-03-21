import torch
import scipy.sparse as sp
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors


def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)

    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))

    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def mixup_data(X, Y, alpha=0.1, mul_factor=2):

    rs = np.random.RandomState(39)
    n = X.shape[0]

    mixed_X = torch.tensor(np.empty((n*(mul_factor-1), X.shape[1]))).cuda()
    mixed_Y = torch.tensor(np.empty(n*(mul_factor-1))).cuda()

    for i in range(mul_factor-1):

        # sample more than needed as some will be filtered out
        lam = np.random.beta(alpha, alpha, size=round(n*2))

        # original data vectors will be concatenated later
        lam = lam[(lam!=0.0) & (lam!=1.0)][:n][:, None]  # shape nx1

        shuffle_idx = rs.choice(np.arange(n), n, replace=False)

        mixed_X[i*n : (i+1)*n] = torch.tensor(lam).cuda() * X + (1 - torch.tensor(lam).cuda()) * X[shuffle_idx, :]
        mixed_Y[i*n : (i+1)*n] = torch.mul(torch.tensor(np.squeeze(lam)).cuda(), Y) + torch.mul((1 - torch.tensor(np.squeeze(lam)).cuda()), Y[shuffle_idx])

    # concatenate original data vectors
    # mixed_X = np.append(mixed_X, X, axis=0)
    # mixed_Y = np.append(mixed_Y, Y, axis=0)

    return mixed_X, mixed_Y


def get_tail_label(df):
    """
    Give tail label colums of the given target dataframe

    args
    df: pandas.DataFrame, target label df whose tail label has to identified

    return
    tail_label: list, a list containing column name of all the tail label
    """
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl) / irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label



def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance

    args
    X: np.array, array whose nearest neighbor has to find

    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices


def MLSMOTE(X, y, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm

    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample

    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.DataFrame):
        y = pd.get_dummies(np.array(y))

    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n - 1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbour, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X.values, np.argmax(target.values,axis=1)


def balance_MLSMOTE(labeled_X, labeled_y, n_sample):

    X_list = []
    y_list = []
    for i in range(max(labeled_y) + 1):
        X_list.append(labeled_X[labeled_y == i, :])
        y_list.append(labeled_y[labeled_y == i])

    num_classes = max(labeled_y) + 1
    one_hot_codes = np.eye(num_classes)

    df_y_list = []
    for i in range(len(y_list)):
        one_hot_labels = []
        for label in y_list[i]:
            one_hot_label = one_hot_codes[label]
            one_hot_labels.append(one_hot_label)
        df_y = pd.DataFrame(np.array(one_hot_labels))
        df_y_list.append(df_y)

    if n_sample == None:
        smote_num = 0
        for i in range(len(y_list)):
            if len(y_list[i]) > smote_num:
                smote_num = len(y_list[i])
                majority_class = i
    else:
        smote_num = n_sample

    for i in range(len(y_list)):
        if smote_num - len(y_list[i]) > 0:
            X_res, y_res = MLSMOTE(X_list[i], df_y_list[i], smote_num - len(y_list[i]))
        else:
            X_res, y_res = X_list[i], y_list[i]
        if i == 0:
            X_smo = X_res
            y_smo = y_res
        else:
            X_smo = np.concatenate([X_smo, X_res], axis=0)
            y_smo = np.concatenate([y_smo, y_res], axis=0)
    return X_smo, np.squeeze(y_smo)