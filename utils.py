import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn import metrics
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F

def random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def calculate_score(label, output):
    auc = metrics.roc_auc_score(label, output)
    precision, recall, _ = metrics.precision_recall_curve(label, output)
    aupr = metrics.auc(recall, precision)
    pred = np.where(output > 0.5, 1, 0)

    accuracy = metrics.accuracy_score(label, pred)
    recall = metrics.recall_score(label, pred)
    precision = metrics.precision_score(label, pred)
    f1_score = metrics.f1_score(label, pred)
    return auc, aupr, accuracy, recall, precision, f1_score


def calculate_auc_aupr(output, label, mask, neg_mask):
    mask_sum = mask + neg_mask
    output = output.reshape(-1)[np.where(mask_sum.reshape(-1) == 1)]
    label = label.reshape(-1)[np.where(mask_sum.reshape(-1) == 1)]
    auc = metrics.roc_auc_score(label, output)
    precision, recall, _ = metrics.precision_recall_curve(label, output)
    aupr = metrics.auc(recall, precision)
    return auc, aupr


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=6.0, reduction='mean', enabled=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.enabled = enabled

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        if self.enabled:
            pt = torch.exp(-bce)
            loss = self.alpha * (1 - pt) ** self.gamma * bce
        else:
            loss = bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(directory+'/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(np.interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AUPR: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AUPR: %.4f $\pm$ %.4f' % (mean_prc, prc_std))

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def load_data(data_dir, k_index):
    rna_features = np.load(data_dir + 'rna_features.npy')
    drug_features = np.load(data_dir + 'drug_features.npy')
    inter_features_rna = np.load(data_dir + 'inter_features_rna.npy')
    inter_features_drug = np.load(data_dir + 'inter_features_drug.npy')
    adj = np.load(data_dir + 'adj.npy')
    interaction = np.load(data_dir + 'interaction.npy')

    coo_inter = coo_matrix(interaction)
    pos_data = np.hstack((coo_inter.row[:, np.newaxis], coo_inter.col[:, np.newaxis]))
    neg_data = np.array(random.choices(np.vstack(np.where(interaction == 0)).transpose(), k=pos_data.shape[0]))

    skf = KFold(n_splits=5, shuffle=True)
    for fold_index, (train_index, test_index) in enumerate(skf.split(pos_data)):
            train_data, test_data = pos_data[train_index], pos_data[test_index]
            train_neg_data, test_neg_data = neg_data[train_index], neg_data[test_index]
    return adj, interaction, rna_features, drug_features, inter_features_rna, inter_features_drug, train_data, test_data, train_neg_data, test_neg_data