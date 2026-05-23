import os
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import from_networkx


class ProGKDGraph(object):

    def __init__(self):
        pass

    def ten_fold_five_crs_validation(self, file_save_path, K=1, folds=5):
        # load k set label.
        final_gene_node, _ = self.get_node_genelist()

        label_file = pd.read_csv("./data_container/Label_gene_GLIMS_intraction_uniprot.csv")
        genes_match = pd.merge(pd.Series(sorted(final_gene_node), name='Hugosymbol'), label_file, on='Hugosymbol', how='left')

        idx_list = np.array(genes_match[~genes_match['Label'].isnull()].index)
        print(f'The match number of gene with annotation: {len(idx_list)}')
        label_list = np.array(genes_match['Label'].loc[idx_list])
        unique, counts = np.unique(label_list, return_counts=True)
        print('The label distribution:', dict(zip(unique, counts)))

        k_sets = {}
        for i in range(K):
            kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            splits = kf.split(idx_list, label_list)

            k_folds = []
            for train, val in splits:
                train_mask = torch.LongTensor(idx_list[train])
                val_mask = torch.LongTensor(idx_list[val])
                train_label = torch.FloatTensor(label_list[train]).reshape(-1, 1)
                val_label = torch.FloatTensor(label_list[val]).reshape(-1, 1)
                k_folds.append((train_mask, val_mask, train_label, val_label))

            k_sets[i] = k_folds

        torch.save(k_sets, os.path.join(file_save_path, 'k_sets.pkl'))

        return k_sets, idx_list, label_list

    def ten_fold_five_crs_validation_psudo_label(self, file_save_path, K=1, folds=5):
        # load k set label.
        final_gene_node, _ = self.get_node_genelist()

        label_file = pd.read_csv("./data_container/Label_gene_GLIMS_intraction_uniprot.csv")
        genes_match = pd.merge(pd.Series(sorted(final_gene_node), name='Hugosymbol'), label_file, on='Hugosymbol', how='left')

        psudo_real_gene = pd.read_csv("./data_container/psudo_and_real_label_gene.csv")

        idx_list = np.array(genes_match[~genes_match['Label'].isnull()].index)
        print(f'The match number of gene with annotation: {len(idx_list)}')
        label_list = np.array(genes_match['Label'].loc[idx_list])
        unique, counts = np.unique(label_list, return_counts=True)
        print('The label distribution:', dict(zip(unique, counts)))

        k_sets = {}

        for i in range(K):
            kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            splits = kf.split(idx_list, label_list)

            k_folds = []
            for train, val in splits:
                mask_truth_idx = torch.LongTensor(idx_list[train])
                train_mask = torch.LongTensor(np.array(psudo_real_gene.drop(idx_list[val]).index))
                val_mask = torch.LongTensor(idx_list[val])
                mask_truth_label = torch.FloatTensor(label_list[train]).reshape(-1, 1)
                train_label = torch.FloatTensor(psudo_real_gene.drop(idx_list[val])['Label'].values).reshape(-1, 1)
                val_label = torch.FloatTensor(label_list[val]).reshape(-1, 1)
                k_folds.append((train_mask, val_mask, train_label, val_label, mask_truth_idx, mask_truth_label))

            k_sets[i] = k_folds

        torch.save(k_sets, os.path.join(file_save_path, 'k_sets.pkl'))

        return k_sets, idx_list, label_list

################################# final_gene_node for gene witout gene deleted ###############################
    def get_node_genelist(self):
        print('Get gene list')


        ### final_gene_node --------
        nodes_df = pd.read_csv("./data_container/gene_name_GLIMS_pancancer.csv")
        gene_uniprot = pd.read_csv("./data_container//gene_embedding19255.csv")

        final_gene_node = sorted(set(gene_uniprot['Unnamed: 0'].values).intersection(set(nodes_df.values.reshape(15796,))))

        genelist0 = []
        with open("./data_container/genesequenceDel.txt", 'r') as f:
            for i in f.read().split(','):
                genelist0.append(i.strip(' \' '))

        gene0 = pd.DataFrame(genelist0)
        final_gene_node = [gene for gene in final_gene_node if gene not in gene0.values]
        #### ----------------------

        final_gene_node_idx = nodes_df[nodes_df['1'].isin(final_gene_node)].index.tolist()

        return final_gene_node, final_gene_node_idx
################################################################################################################


    def get_node_omic_embedding_feature(self):

        final_gene_node, _ = self.get_node_genelist()

        # process the omic data
        expr_df = pd.read_csv("./data_container/expr_gene_GLIMS_intraction_uniprot.csv").drop('Hugosymbol', axis=1)
        mut_df = pd.read_csv("./data_container/mut_gene_GLIMS_intraction_uniprot.csv").drop('Hugosymbol', axis=1)
        cn_df = pd.read_csv("./data_container/cn_gene_GLIMS_intraction_uniprot.csv").drop('Hugosymbol', axis=1)
        methy_df = pd.read_csv("./data_container/methy_gene_GLIMS_intraction_uniprot.csv")
        omics_data = pd.concat([expr_df, mut_df, cn_df, methy_df], axis=1)
        omics_data = omics_data.sort_values(by=['Hugosymbol'])
        omics_data = omics_data.drop('Hugosymbol', axis=1)



        omics_feature_vector = sp.csr_matrix(omics_data, dtype=np.float32)
        omics_feature_vector = torch.FloatTensor(
            np.array(omics_feature_vector.todense()))
        print(f'The shape of omics_feature_vector:{omics_feature_vector.shape}')

        # process the embedding data
        gene_embedding = pd.read_csv("./data_container/protT5GLIMS_gene_embedding.csv", index_col=0)

        emb_feature_vector = sp.csr_matrix(gene_embedding, dtype=np.float32)
        emb_feature_vector = torch.FloatTensor(
            np.array(emb_feature_vector.todense()))
        emb_feature_vector = emb_feature_vector.unsqueeze(1)

        return omics_feature_vector, emb_feature_vector, final_gene_node

    def generate_graph(self):
        """
             generate graph
        """
        print('generate graph')
        ppi = pd.read_csv("./data_container/network_gene_GLIMS_intraction_uniprot_pancancer.csv", index_col=0)
        return ppi


    def load_featured_graph(self, network, omicfeature):
        omics_feature_vector = sp.csr_matrix(omicfeature, dtype=np.float32)
        omics_feature_vector = torch.FloatTensor(np.array(omics_feature_vector.todense()))
        print(f'The shape of omics_feature_vector:{omics_feature_vector.shape}')

        if network.shape[0] == network.shape[1]:
            G = nx.from_pandas_adjacency(network)
        else:
            G = nx.from_pandas_edgelist(network)

        G_adj = nx.convert_node_labels_to_integers(G, ordering='sorted', label_attribute='label')

        print(f'If the graph is connected graph: {nx.is_connected(G_adj)}')
        print(f'The number of connected components: {nx.number_connected_components(G_adj)}')

        graph = from_networkx(G_adj)
        assert graph.is_undirected() == True
        print(f'The edge index is {graph.edge_index}')

        graph.x = omics_feature_vector

        return graph
