import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.sparse as sp
import torch
from torch.nn import BCELoss
import numpy as np
from dim_vis import plot_pca_tsne_umap



from HDIGRL import *
from utils import *

seed = 42
random_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_fold_results = []


def train_fold(model, train_data, train_label, test_data, test_label, adj,
               rna_feat, drug_feat, inter_rna, inter_drug, epochs, optimizer, loss_fn):
    train_losses = []
    train_aucs = []
    test_losses = []
    test_aucs = []

    for epoch in range(epochs):
        model.train()
        model.zero_grad()

        x = model.projection_and_aggregation(rna_feat, drug_feat, inter_rna, inter_drug)

        train_output = model(x, adj, train_data)
        train_loss = loss_fn(train_output, train_label)


        train_auc = metrics.roc_auc_score(
            train_label.detach().cpu().numpy().reshape(-1),
            train_output.detach().cpu().numpy().reshape(-1)
        )


        train_loss.backward()
        optimizer.step()


        model.eval()
        with torch.no_grad():
            x = model.projection_and_aggregation(rna_feat, drug_feat, inter_rna, inter_drug)
            test_output = model(x, adj, test_data)
            test_loss = loss_fn(test_output, test_label)
            test_auc = metrics.roc_auc_score(
                test_label.detach().cpu().numpy().reshape(-1),
                test_output.detach().cpu().numpy().reshape(-1)
            )


        train_losses.append(train_loss.item())
        train_aucs.append(train_auc)
        test_losses.append(test_loss.item())
        test_aucs.append(test_auc)


        if (epoch + 1) % 50 == 0:
            print(f'Fold {current_fold + 1}/5, Epoch {epoch + 1}/{epochs} - '
                  f'Train Loss: {train_loss.item():.6f}, Train AUC: {train_auc:.6f}, '
                  f'Test Loss: {test_loss.item():.6f}, Test AUC: {test_auc:.6f}')


    model.eval()
    with torch.no_grad():

        x = model.projection_and_aggregation(rna_feat, drug_feat, inter_rna, inter_drug)


        final_output = model(x, adj, test_data)


        final_auc, final_aupr, final_acc, final_recall, final_precision, final_f1 = calculate_score(
            test_label.detach().cpu().numpy(),
            final_output.detach().cpu().numpy()
        )

        fpr, tpr, _ = roc_curve(
            test_label.detach().cpu().numpy(),
            final_output.detach().cpu().numpy()
        )


        test_embed = model.embedding(x, adj, test_data)
        features_np = test_embed.detach().cpu().numpy()
        labels_np = test_label.detach().cpu().numpy().reshape(-1).astype(int)


    plot_pca_tsne_umap(features_np, labels_np, save_path="pca_tsne_umap.png")

    return {
        'train_losses': train_losses,
        'train_aucs': train_aucs,
        'test_losses': test_losses,
        'test_aucs': test_aucs,
        'final_auc': final_auc,
        'final_aupr': final_aupr,
        'final_acc': final_acc,
        'final_recall': final_recall,
        'final_f1': final_f1,
        'fpr': fpr,
        'tpr': tpr,
        'y_true': test_label.detach().cpu().numpy(),
        'y_pred': final_output.detach().cpu().numpy()
    }


def plot_cv_metrics():
    fprs = [fold['fpr'] for fold in all_fold_results]
    tprs = [fold['tpr'] for fold in all_fold_results]
    aucs = [fold['final_auc'] for fold in all_fold_results]
    auprs = [fold['final_aupr'] for fold in all_fold_results]
    accs = [fold['final_acc'] for fold in all_fold_results]
    recalls = [fold['final_recall'] for fold in all_fold_results]
    f1s = [fold['final_f1'] for fold in all_fold_results]

    print('\n五折交叉验证结果汇总:')
    print(f'平均AUC: {np.mean(aucs):.6f} ± {np.std(aucs):.6f}')
    print(f'平均AUPR: {np.mean(auprs):.6f} ± {np.std(auprs):.6f}')
    print(f'平均准确率: {np.mean(accs):.6f} ± {np.std(accs):.6f}')
    print(f'平均召回率: {np.mean(recalls):.6f} ± {np.std(recalls):.6f}')
    print(f'平均F1值: {np.mean(f1s):.6f} ± {np.std(f1s):.6f}')


    plot_auc_curves(fprs, tprs, aucs, './', 'cv_roc_curve')


    plt.figure(figsize=(15, 10))


    plt.subplot(2, 1, 1)
    for i, fold in enumerate(all_fold_results):
        plt.plot(range(1, len(fold['train_losses']) + 1),
                 fold['train_losses'], label=f'Fold {i + 1} Train', alpha=0.6)
        plt.plot(range(1, len(fold['test_losses']) + 1),
                 fold['test_losses'], label=f'Fold {i + 1} Test', alpha=0.6)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Across Folds')
    plt.legend()


    plt.subplot(2, 1, 2)
    for i, fold in enumerate(all_fold_results):
        plt.plot(range(1, len(fold['train_aucs']) + 1),
                 fold['train_aucs'], label=f'Fold {i + 1} Train', alpha=0.6)
        plt.plot(range(1, len(fold['test_aucs']) + 1),
                 fold['test_aucs'], label=f'Fold {i + 1} Test', alpha=0.6)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training and Test AUC Across Folds')
    plt.legend()

    plt.tight_layout()
    plt.savefig('cv_metrics_curves.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='./data/')
    parser.add_argument("--num-epochs", type=int, default=600)
    parser.add_argument("--hid-r", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-features", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    args = parser.parse_args()


    adj, interaction, rna_features, drug_features, inter_features_rna, inter_features_drug, _, _, _, _ = load_data(
        data_dir=args.data_dir, k_index=0)

    coo_inter = coo_matrix(interaction)
    pos_data = np.hstack((coo_inter.row[:, np.newaxis], coo_inter.col[:, np.newaxis]))
    neg_data = np.array(random.choices(
        np.vstack(np.where(interaction == 0)).transpose(),
        k=pos_data.shape[0]
    ))


    skf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for current_fold, (train_idx, test_idx) in enumerate(skf.split(pos_data)):
        print(f'\n===== 开始第 {current_fold + 1}/5 折训练 =====')


        train_pos = pos_data[train_idx]
        test_pos = pos_data[test_idx]
        train_neg = neg_data[train_idx]
        test_neg = neg_data[test_idx]


        train_data = np.vstack([train_pos, train_neg])
        train_label = torch.tensor(
            np.vstack([np.ones([train_pos.shape[0], 1]),
                       np.zeros([train_neg.shape[0], 1])]),
            dtype=torch.float32).to(device)

        test_data = np.vstack([test_pos, test_neg])
        test_label = torch.tensor(
            np.vstack([np.ones([test_pos.shape[0], 1]),
                       np.zeros([test_neg.shape[0], 1])]),
            dtype=torch.float32).to(device)





        global_node_num = int(adj.shape[0] * 0.1)
        adj_expanded = np.vstack((
            np.hstack((adj, np.ones((adj.shape[0], global_node_num)))),
            np.hstack((np.ones((global_node_num, adj.shape[0])),
                       np.zeros((global_node_num, global_node_num))))
        ))
        sp_adj = sp.coo_matrix(adj_expanded)
        adj_tensor = torch.LongTensor(np.vstack((sp_adj.row, sp_adj.col))).to(device)


        rna_feat = torch.tensor(rna_features).to(device)
        drug_feat = torch.tensor(drug_features).to(device)
        inter_rna = torch.tensor(inter_features_rna).to(device)
        inter_drug = torch.tensor(inter_features_drug).to(device)


        model = HDIGRL(
            r=args.hid_r,
            n_layers=args.n_layers,
            n_features=args.n_features,
            num_rna=rna_features.shape[0],
            num_dis=drug_features.shape[0],
            n_global_node=global_node_num
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = FocalLoss(alpha=0.9, gamma=6, enabled=True)



        fold_results = train_fold(
            model=model,
            train_data=train_data,
            train_label=train_label,
            test_data=test_data,
            test_label=test_label,
            adj=adj_tensor,
            rna_feat=rna_feat,
            drug_feat=drug_feat,
            inter_rna=inter_rna,
            inter_drug=inter_drug,
            epochs=args.num_epochs,
            optimizer=optimizer,
            loss_fn=loss_fn
        )

        ckpt_path = f"ckpt_fold{current_fold + 1}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args)
        }, ckpt_path)
        print(f"[Saved] {ckpt_path}")

        all_fold_results.append(fold_results)

    plot_cv_metrics()