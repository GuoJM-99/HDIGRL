import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


input_dir = './raw_data/'
output_dir = './data/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if __name__ == '__main__':
    association = pd.read_csv(input_dir + 'association.txt', sep='\t', header=None).loc[:, :1]
    drug = pd.read_csv(input_dir + 'drug.txt', sep='\t').loc[:, ['id', 'drug']]
    rna = pd.read_csv(input_dir + 'miRNA.txt', sep='\t').loc[:, ['id', 'miRNAname']]

    interaction = np.zeros((431, 140))
    errCount = 0
    for index, row in association.iterrows():
        drug_name = row[0]
        miRNA_name = row[1]
        try:
            x = np.where(rna == miRNA_name)[0][0]
            y = np.where(drug == drug_name)[0][0]
            interaction[x][y] = 1
        except IndexError as e:
            errCount += 1
    d_d = pd.read_csv(input_dir + 'd-d.txt', sep=',')
    m_m = pd.read_csv(input_dir + 'm-m.txt', sep=',')

    ddInteraction = np.identity(140)
    mmInteraction = np.identity(431)
    for index, row in d_d.iterrows():
        x = row.iloc[0]
        y = row.iloc[1]
        ddInteraction[x][y] = 1
        ddInteraction[y][x] = 1
    for index, row in m_m.iterrows():
        x = row.iloc[0]
        y = row.iloc[1]
        mmInteraction[x][y] = 1
        mmInteraction[y][x] = 1

    adj = np.vstack((np.hstack((mmInteraction, interaction)), np.hstack((interaction.transpose(), ddInteraction))))
    np.save(output_dir + "interaction.npy", interaction.astype(np.float32))
    np.save(output_dir + "adj.npy", adj.astype(np.float32))

    gamma_drug = 1 / (np.sum(np.linalg.norm(interaction, ord=2, axis=0) ** 2) / interaction.shape[1])
    disease_interaction = np.zeros((drug.shape[0], drug.shape[0]))
    for i in range(disease_interaction.shape[0]):
        for j in range(disease_interaction.shape[1]):
            disease_interaction[i, j] = np.exp(
                -gamma_drug * (np.linalg.norm(interaction[:, i] - interaction[:, j], ord=2) ** 2))

    gamma_rna = 1 / (np.sum(np.linalg.norm(interaction, ord=2, axis=1) ** 2) / interaction.shape[0])
    rna_interaction = np.zeros((rna.shape[0], rna.shape[0]))
    for i in range(rna_interaction.shape[0]):
        for j in range(rna_interaction.shape[1]):
            rna_interaction[i, j] = np.exp(
                -gamma_rna * (np.linalg.norm(interaction[i, :] - interaction[j, :], ord=2) ** 2))

    coo_inter = coo_matrix(interaction)
    coo_file = np.hstack((coo_inter.row[:, np.newaxis], coo_inter.col[:, np.newaxis], coo_inter.data[:, np.newaxis]))

    G_RNA = nx.from_numpy_array(mmInteraction - np.identity(mmInteraction.shape[0]))
    rna_features = np.copy(mmInteraction)
    for u, v, p in nx.adamic_adar_index(G_RNA):
        rna_features[u, v] = p
        rna_features[v, u] = p

    G_DRUG = nx.from_numpy_array(ddInteraction - np.identity(ddInteraction.shape[0]))
    drug_features = np.copy(ddInteraction)
    for u, v, p in nx.adamic_adar_index(G_DRUG):
        drug_features[u, v] = p
        drug_features[v, u] = p

    inter_features_drug = np.zeros((drug.shape[0], drug.shape[0]))
    for i in range(inter_features_drug.shape[0]):
        for j in range(inter_features_drug.shape[1]):
            inter_features_drug[i, j] = 1 / (abs(interaction[:, i] - interaction[:, j]).sum() + 1)

    inter_features_rna = np.zeros((rna.shape[0], rna.shape[0]))
    for i in range(inter_features_rna.shape[0]):
        for j in range(inter_features_rna.shape[1]):
            inter_features_rna[i, j] = 1 / (abs(interaction[i, :] - interaction[j, :]).sum() + 1)


    np.save(output_dir + "rna_features.npy", rna_features.astype(np.float32))
    np.save(output_dir + "drug_features.npy", drug_features.astype(np.float32))
    np.save(output_dir + "inter_features_rna.npy", inter_features_rna.astype(np.float32))
    np.save(output_dir + "inter_features_drug.npy", inter_features_drug.astype(np.float32))
